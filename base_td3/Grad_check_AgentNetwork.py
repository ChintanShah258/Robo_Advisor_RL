import numpy as np
import torch as T
import torch.nn.functional as F
from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
import os
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR

T.autograd.set_detect_anomaly(True)


class Agent():
    def __init__(self,alpha,beta,input_dims,tau,env,gamma=0.99,update_actor_interval = 5,warm_up=1000,
                 n_actions=8,n_assets=None,max_size=200000, layer1_size=400,layer2_size=300,
                 batch_size=100,noise=0.1, T_max = 864, eta_min = 1e-6,vol_bias=None):
        
        self.gamma = gamma
        self.tau = tau
        self.T_max = T_max
        self.eta_min = eta_min
        #Since we are adding noise to the fully determinitic policy netwrok we need to makre sure '
        # they are constrained within the environment. So we are keeping track for max and min
        # actions for our environment. For our problem we would have the constraints that action values
        # all add up to 1 (i.e. portfolio weights) with no negative values (or short selling) 
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.memory = ReplayBuffer(max_size,input_dims,n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warm_up = warm_up
        self.n_actions = n_actions
        self.n_assets = n_assets
        self.update_actor_interval = update_actor_interval
        self.current_episode = 0
        
        self.actor = ActorNetwork(alpha, input_dims, layer1_size,layer2_size, n_assets=n_assets,
                                  name='actor', vol_bias=vol_bias,hist_window=env.hist_window)
        self.critic_1 = CriticNetwork(beta, input_dims, layer1_size,layer2_size, n_actions=n_actions,name='critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, layer1_size,layer2_size, n_actions=n_actions,name='critic_2')
        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size,layer2_size, n_assets=n_assets,
                                         name='target_actor', vol_bias=vol_bias,hist_window=env.hist_window)
        self.target_critic_1 = CriticNetwork(beta, input_dims, layer1_size,layer2_size, n_actions=n_actions,name='target_critic_1') 
        self.target_critic_2 = CriticNetwork(beta, input_dims, layer1_size,layer2_size, n_actions=n_actions,name='target_critic_2')
        
        # Keep track of noise for the exploration portion of the exploration-exploitation dilemma
        self.noise = noise
        
        # ── here‘s the new bit: attach cosine‐annealing schedulers ──
        # choose T_max = total number of scheduler steps per cycle (e.g. 10k)
        
        
        self.actor_scheduler   = CosineAnnealingLR(self.actor.optimizer,   T_max=self.T_max, eta_min=eta_min)
        self.critic_1_scheduler = CosineAnnealingLR(self.critic_1.optimizer, T_max=self.T_max, eta_min=eta_min)
        self.critic_2_scheduler = CosineAnnealingLR(self.critic_2.optimizer, T_max=self.T_max, eta_min=eta_min)

        # We call the update network parameters function. We do that so that we can set the  
        # intial weights of the target network equal to the exact weights of the online networks 
        # tau is the parameter of the update. tau=1 means the same update     
        self.update_network_parameters(tau=0.007)
        
    def choose_action(self, observation): #Obervation of the current state of the env as input
        if self.time_step < self.warm_up:
            pi = T.randn(1, self.n_actions, device=self.actor.device) * self.noise
            # need to pass the size(shape) of the tensor as a tuple, so
            # self.n_actions, makes it a single-element tuple
            # We might have to pass something like.  
            # size = (self.n_actions + self.lambda_value,)
        else:
            #If we are outside of the warm-up period then we want to find a deterministic action
            state = T.tensor(observation,dtype = T.float).to(self.actor.device)
            pi = self.actor.forward(state).to(self.actor.device)
            # print(f"[AGENT_PRE_NOISE] pi: {pi.detach().cpu().numpy()}")
        #Added extra noise. According to the coder in warmup period we are exploring so adding this noise
        # shouldn't matter, but it is not necessary
        
        #This is a common noise for all the actions
        #pi_prime_noisy = pi+T.tensor(np.random.normal(scale=self.noise),dtype=T.float).to(self.actor.device)
        
        noise_tensor   = T.randn_like(pi) * self.noise
        #Adding in noise for each of the action instead of one common noise for all actions 
        pi_prime_noisy = pi + noise_tensor
        # print(f"[AGENT_POST_NOISE_PI] pi_prime_noisy: {pi_prime_noisy.detach().cpu().numpy()},shape: {pi_prime_noisy.shape}")
        #Since this environment is bounded between (-1,1) we need to clamp the actions
        pi_prime_noisy = T.clamp(pi_prime_noisy,self.min_action[0],self.max_action[0])
        
        # 2) slice back out
        n = self.n_assets
        raw_base   = pi_prime_noisy[:, :n].squeeze(0)        # shape (n_assets,)
        #print(f"[AGENT_POST_NOISE] raw_base:", raw_base.detach().cpu().numpy(),
            #"sum:", raw_base.sum(dim=0).item(),"shape:", raw_base.shape)
        raw_risky  = pi_prime_noisy[:, n:2*n].squeeze(0)
        #print(f"[AGENT_POST_NOISE] raw_risky:", raw_risky.detach().cpu().numpy(),
            #"sum:", raw_risky.sum(dim=0).item())
        raw_lambda = pi_prime_noisy[:, 2*n:2*n+1].squeeze(0) # shape (1,)
        #print(f"[AGENT_POST_NOISE] raw_lambda:", raw_lambda.detach().cpu().numpy())
        raw_theta  = pi_prime_noisy[:, 2*n+1:2*n+2].squeeze(0)
        #print(f"[AGENT_POST_NOISE] raw_theta:", raw_theta.detach().cpu().numpy())

        # 3) re‑apply activations
        w_base       = F.softmax(raw_base.unsqueeze(0), dim=-1).squeeze(0)
        #print(f"[AGENT_POST_NOISE] w_base:", w_base.detach().cpu().numpy(),
            #"sum:", w_base.sum(dim=0).item())
        w_risky      = F.softmax(raw_risky.unsqueeze(0), dim=-1).squeeze(0)
        #print(f"[AGENT_POST_NOISE] w_risky:", w_risky.detach().cpu().numpy(),
            #"sum:", w_risky.sum(dim=0).item())
        amount_risky = T.sigmoid(raw_lambda)
        theta_risky  = T.sigmoid(raw_theta)

        # 4) recombine into final action
        pi_prime = T.cat([w_base, w_risky, amount_risky, theta_risky], dim=0)
        
        self.time_step += 1
        
        pi_prime = pi_prime.cpu().detach().numpy()  # shape (action_dim,) when B=1
        # ensure a batch dimension
        if pi_prime.ndim == 1:
            pi_prime = pi_prime[None, :]   # now shape (1, action_dim)
        return pi_prime
        #Convoluted dereferencing due to the way Pytorch handles tensors. We passed a numpy array into the
        # environment for an action
        
    #Interfacing Function to store transition in the network's memory
    def remember(self, state, action, reward, new_state,done):
        self.memory.store_transition(state, action, reward, new_state,done)
        
    # This determines whether we wait for the Replay Buffer to be filled before we start learning or not
    # Replay Buffer is initialized with 0s. Here we opting to start the learning as soon as the buffer is 
    # filled up to the batch_size or you let the agent play enough times to fill its replay memory before 
    # you start learning. Here the memory size seems huge so we start learning once we hit the batch_size.
    
        
    def learn(self):
        """
        Perform one learning step (TD3 update) and return the batch critic and actor losses.
        Returns:
            (critic_loss, actor_loss) as Python floats (actor_loss may be None if no actor update)
        """
        # Skip until we have enough samples
        
        if self.memory.mem_cntr < self.batch_size:
            #print(f"[AGENT_DEBUG] mem_cntr={self.memory.mem_cntr}, batch_size={self.batch_size}")
            return None, None
        
        # 🛠 DEBUG: exactly when we cross the threshold
        if self.memory.mem_cntr == self.batch_size and self.learn_step_cntr == 0:
            print("🎉 [AGENT_DEBUG] buffer filled, now training kicks in")

        # Sample a batch from replay buffer
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        # Convert to tensors on the correct device
        device = self.critic_1.device
        reward = T.tensor(reward, dtype=T.float, device=device).view(-1, 1)
        done   = T.tensor(done,   dtype=T.bool,  device=device)
        state_ = T.tensor(new_state, dtype=T.float, device=device)
        state  = T.tensor(state,     dtype=T.float, device=device)
        action = T.tensor(action,    dtype=T.float, device=device)
        
            # ── INSERT DIAGNOSTIC HERE ──
        # 1) Find which batch entries have negative r_env
        with T.no_grad():
            r_np = reward.cpu().numpy().flatten()
        neg_idx = np.where(r_np < 0)[0]
        print("Batch r_env:", np.round(r_np,4))
        print("Negative indices:", neg_idx.tolist())

        # 2) If any negative, isolate them
        mask = (reward.view(-1) < 0)
        if mask.any():
            state_neg  = state[mask]
            # compute actor‐gradient on only those “bad” states
            actor_q_neg = self.critic_1(state_neg, self.actor(state_neg))
            actor_loss_neg = -actor_q_neg.mean()
            # zero existing grads
            self.actor.optimizer.zero_grad()
            actor_loss_neg.backward()
            print("⧉ Actor gradients on negative-r_env samples:")
            for name, p in self.actor.named_parameters():
                if p.grad is not None:
                    print(f"  {name:30s} grad_norm = {p.grad.norm().item():.6f}")
            # clear those grads so they don’t actually get applied now
            self.actor.optimizer.zero_grad()
            # ── end diagnostic ──

        # Compute target actions with smoothing
        target_actions = self.target_actor(state_)
        # 1) add noise & clamp (you already have this)
        noise_tensor   = T.randn_like(target_actions) * self.noise
        target_actions = T.clamp(target_actions + noise_tensor, self.min_action[0], self.max_action[0])

        # 2) slice into the four parts
        n = self.n_assets
        raw_base    = target_actions[:,     :n]
        raw_risky   = target_actions[:,  n:2*n]
        raw_lambda  = target_actions[:,2*n:2*n+1]
        raw_theta   = target_actions[:,2*n+1:2*n+2]

        # 3) re-apply your activations
        w_base       = F.softmax(raw_base,  dim=1)
        w_risky      = F.softmax(raw_risky, dim=1)
        lambda_t     = T.sigmoid(raw_lambda)
        theta_t      = T.sigmoid(raw_theta)

        # 4) recombine
        target_actions = T.cat([w_base, w_risky, lambda_t, theta_t], dim=1)

        # Compute target Q-values
        q1_target = self.target_critic_1(state_, target_actions)
        q2_target = self.target_critic_2(state_, target_actions)
        q_target   = T.min(q1_target, q2_target)
        q1_target = q1_target.masked_fill(done, 0.0)
        q2_target = q2_target.masked_fill(done, 0.0)

        # Critic loss
        target_q = reward + self.gamma * q_target
        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        critic_loss = q1_loss + q2_loss

        # Backward pass for critics
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        # --- diagnostic: measure critic gradient w.r.t. λ head ---
        # a_curr is the actor’s output for the batch
        a_curr = self.actor(state)
        a_curr.requires_grad_(True)

        q_val = self.critic_1(state, a_curr)    # shape (B,1)
        grads  = T.autograd.grad(q_val.mean(), a_curr)[0]  # shape (B,action_dim)

        λ_index = 2 * self.n_assets
        print("dQ/dλ mean:", grads[:, λ_index].mean().item())
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_1_scheduler.step()
        self.critic_2.optimizer.step()
        self.critic_2_scheduler.step()
        
        # Log critic stats
        wandb.log({
            "critic/step":     self.learn_step_cntr,
            "critic/loss":     critic_loss.item(),
            "critic/mean_q":   current_q1.mean().item(),
            "critic/target_q": q1_target.mean().item(),
        }, step=self.learn_step_cntr, commit=False)

        # Track actor loss if updated
        actor_loss_item = None

        # Update actor every `update_actor_interval` steps
        if self.learn_step_cntr % self.update_actor_interval == 0:
            # Actor loss (maximizing Q1)
            self.actor.optimizer.zero_grad()
            actor_q = self.critic_1(state, self.actor(state))
            actor_loss = -actor_q.mean()
            actor_loss.backward()
            self.actor.optimizer.step()
            self.actor_scheduler.step()
            
            # Log actor stats
            wandb.log({
                "critic/step": self.learn_step_cntr,
                "actor/loss": actor_loss.item(),
            }, step=self.learn_step_cntr, commit=False)

            actor_loss_item = actor_loss.item()

            # Soft update target networks
            self.update_network_parameters()

        # Increment global learning counter
        self.learn_step_cntr += 1

        return critic_loss.item(), actor_loss_item

    
    def update_network_parameters(self,tau=None):
        if tau is None:
            tau = self.tau #this will take care of the corener case
        
        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()
        
        critic_1 = dict(critic_1_params)
        critic_2 = dict(critic_2_params)
        actor = dict(actor_params)
        target_actor = dict(target_actor_params)
        target_critic_1 = dict(target_critic_1_params)
        target_critic_2 = dict(target_critic_2_params)
        
        #Iterating over the above dictionaries
        for name in critic_1:
            critic_1[name] = tau*critic_1[name].clone() + (1-tau)*target_critic_1[name].clone()
        
        for name in critic_2:
            critic_2[name] = tau*critic_2[name].clone() + (1-tau)*target_critic_2[name].clone()
    
        for name in actor:
            actor[name] = tau*actor[name].clone() + (1-tau)*target_actor[name].clone()
        
        #Uploading the state dictionary
        self.target_critic_1.load_state_dict(critic_1)
        self.target_critic_2.load_state_dict(critic_2)
        self.target_actor.load_state_dict(actor, strict=False)
        #Strict=False is used to ignore the extra parameters in the actor network
        # that are not present in the target actor network
        # This is because the target actor network is a copy of the actor network
        # and we are not using the extra parameters in the target actor network - like volatility_buffer
        
    def save_models(self, label="latest", current_episode=None):
        """
        Save a complete Agent checkpoint under `agent_{label}.pth`, including:
          - training counters
          - current_episode
          - network weights + optimizers
          - replay buffer
          - RNG states
        """
        ep = current_episode if current_episode is not None else self.current_episode

        ckpt = {
            # 1) training‐state counters
            'time_step':        self.time_step,
            'learn_step_cntr':  self.learn_step_cntr,
            'current_episode':  ep,

            # 2) network weights
            'actor_state':         self.actor.state_dict(),
            'critic1_state':       self.critic_1.state_dict(),
            'critic2_state':       self.critic_2.state_dict(),
            'target_actor_state':  self.target_actor.state_dict(),
            'target_critic1_state':self.target_critic_1.state_dict(),
            'target_critic2_state':self.target_critic_2.state_dict(),

            # 3) optimizer state dicts
            'actor_opt':           self.actor.optimizer.state_dict(),
            'critic1_opt':         self.critic_1.optimizer.state_dict(),
            'critic2_opt':         self.critic_2.optimizer.state_dict(),

            # 4) replay buffer contents (you must implement these)
            #'replay_buffer':       self.memory.save_to_dict(replay_window_size=5000),

            # 5) RNG states for exact reproducibility
            'torch_rng':           T.get_rng_state(),
            'cuda_rngs':           T.cuda.get_rng_state_all(),
            'numpy_rng':           np.random.get_state(),
            # if you use Python’s random: 'python_rng': random.getstate()
        }

        ckpt_dir = self.actor.checkpoint_dir
        fname = os.path.join(ckpt_dir, f"agent_{label}.pth")
        os.makedirs(ckpt_dir, exist_ok=True)
        T.save(ckpt, fname)
        print(f"[Agent] Saved checkpoint '{label}' at episode {ep}: {fname}")


    def load_models(self, label="latest"):
        fname = os.path.join(self.actor.checkpoint_dir, f"agent_{label}.pth")
        if not os.path.isfile(fname):
            print(f"[Agent] No checkpoint found at '{fname}'")
            return

        # load everything (including numpy, lists, etc.)
        ckpt = T.load(fname, map_location="cpu", weights_only=False)

        # 1) counters
        self.time_step       = ckpt['time_step']
        self.learn_step_cntr = ckpt['learn_step_cntr']
        self.current_episode = ckpt.get('current_episode', 0)

        # 2) network weights
        self.actor.load_state_dict(       ckpt['actor_state'])
        self.critic_1.load_state_dict(    ckpt['critic1_state'])
        self.critic_2.load_state_dict(    ckpt['critic2_state'])
        self.target_actor.load_state_dict(       ckpt['target_actor_state'])
        self.target_critic_1.load_state_dict(    ckpt['target_critic1_state'])
        self.target_critic_2.load_state_dict(    ckpt['target_critic2_state'])

        # 3) optimizer states
        self.actor.optimizer.load_state_dict(   ckpt['actor_opt'])
        self.critic_1.optimizer.load_state_dict(ckpt['critic1_opt'])
        self.critic_2.optimizer.load_state_dict(ckpt['critic2_opt'])

        # 4) replay buffer
        #self.memory.load_from_dict(ckpt['replay_buffer'])

        # 5) **only** restore CPU RNG
        cpu_rng = ckpt['torch_rng']
        if cpu_rng.device.type != 'cpu' or cpu_rng.dtype != T.uint8:
            cpu_rng = cpu_rng.to('cpu').to(dtype=T.uint8)
        T.set_rng_state(cpu_rng)

        print(f"[Agent] Loaded checkpoint '{label}' from episode {self.current_episode}")

