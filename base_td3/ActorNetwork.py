import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_assets, hist_window,
                 name, vol_bias: np.ndarray = None,chkpt_dir='base_td3/actor/checkpoints'):
        """
        Args:
            alpha (float): Learning rate.
            input_dims (int): Dimension of the state input.
            fc1_dims (int): Number of units in first fully-connected layer.
            fc2_dims (int): Number of units in second fully-connected layer.
            n_assets (int): Number of n_assets in the asset universe.
            name (str): Name of the network (for checkpointing).
            chkpt_dir (str): Directory to save checkpoints.
            vol_bias: Vector of length n_assets to modulate risky weights.
            
        The composite action will include:
            - w_base: Conservative weights (length n_assets)
            - w_risky: Risky weights (length n_assets)
            - lambda: Fraction of excess capital deployed (scalar, bounded [0,1])
            - theta: Risk modulation parameter for the risky allocation (scalar)
            
        Thus, the total output dimension is: 2*n_assets + 2.
        """
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        print(f"[ActorNetwork] INITIALIZING with input_dims = {input_dims}")
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_assets = n_assets
        self.name = name
        self.hist_window = hist_window
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_base_td3')
        
        # Common trunk that produces a latent representation z.
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.ln1 = nn.LayerNorm(self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.ln2 = nn.LayerNorm(self.fc2_dims)        
        # — register your external volatility‐bias vector —
        # vol_bias should be shape (n_assets,)
        # we store it as a persistent, non-trainable tensor
        if vol_bias is not None:
            # name the buffer “volatility_buffer”
            self.register_buffer('volatility_buffer',
                                 T.tensor(vol_bias, dtype=T.float))
        else:
            # fallback to ones if nothing passed
            self.register_buffer('volatility_buffer',
                                 T.ones(self.n_assets, dtype=T.float))
        
        # Separate output heads:
        # Conservative portfolio weights (w_base): use softmax for valid probability distribution.
        self.base_head = nn.Linear(self.fc2_dims, n_assets)
        # Risky portfolio weights (w_risky): initially output logits, later may be modulated by theta.
        self.risky_head_raw = nn.Linear(self.fc2_dims, n_assets)
        # Degree Risky head: output a scalar controlling fraction of excess capital deployed (bounded 0 to 1).
        self.amount_risky = nn.Linear(self.fc2_dims, 1)
        # Theta head: output a scalar controlling the risk profile for the risky allocation.
        self.theta_head = nn.Linear(self.fc2_dims, 1)
        
        self.optimizer = optim.Adam(
            self.parameters(), lr=alpha, weight_decay=0)   # <-- add a small L2 penalty)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        # ensure state is [batch, features]
        if state.dim() > 2:
            state = state.view(state.size(0), -1)
        elif state.dim() == 1:
            state = state.unsqueeze(0)
        
        # 1) pull out the price-history block from the front of the state
        #    we know prices took up n_assets*hist_window positions:
        ph = state[:, : self.n_assets * self.hist_window ]
        # reshape into (batch, hist_window, n_assets)
        ph = ph.view(-1, self.hist_window, self.n_assets)
        # grab the very latest price
        latest_price = ph[:, -1, :]              # shape (batch, n_assets)
        
        # #Use this for None LayerNorm
        # x = F.relu(self.fc1(state))
        # x = F.relu(self.fc2(x))
        
        #Use these for LayerNorm
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))

        # Conservative portfolio weights
        w_base = T.softmax(self.base_head(x), dim=-1)  # Ensures weights sum to 1
        #print(f"[ACTOR_PRE_NOISE]w_base:", w_base.detach().cpu().numpy(),"  sum:", w_base.sum())
        # Risky portfolio weights
        theta_risky = T.sigmoid(self.theta_head(x))         # θ ∈ (0, 1), can be scaled later if needed
        z_risky = self.risky_head_raw(x)              # Latent pre-risky weights
        w_risky = self.modulate_risky_weights(z_risky, theta_risky, latest_price)

        # Lambda (fraction of excess capital to invest in risky assets)
        amount_risky = T.sigmoid(self.amount_risky(x))   # λ ∈ (0, 1)

        #print(f"[ACTOR_DEBUG]: w_base shape: {w_base.shape}, w_risky shape: {w_risky.shape}, amount_risky shape: {amount_risky.shape}, theta_risky shape: {theta_risky.shape}")
        
        # Combine all outputs into one action vector
        action = T.cat([w_base, w_risky, amount_risky, theta_risky], dim=1)
        
        # # DEBUG:
        # print("w_base:", w_base.detach().cpu().numpy(),
        #     "  sum:", w_base.sum(dim=-1).item())
        # print("w_risky:", w_risky.detach().cpu().numpy(),
        #     "  sum:", w_risky.sum(dim=-1).item())

        return action
    
    def modulate_risky_weights(self, z_risky, theta_risky, latest_price):
        """
        Placeholder function to modulate risky weights based on theta.
        For now, we'll do additive shift: z + theta_risky * c (c is a learnable or constant vector)
        Later you can implement: 
        - Multiplicative: z * (1 + theta)
        - Attention-based risk modulation
        - Nonlinear transformations
        """
        # Example: Additive modulation using a constant risk preference vector
        # (shape matching z_risky, e.g. same number of stocks)
        # volatility_buffer: shape (n_assets,)
        # expand to (batch, n_assets)
        
        vol_bias = self.volatility_buffer.unsqueeze(0)
        
        #print(f"vol_bias shape: {vol_bias.shape}, z_risky shape: {z_risky.shape}, theta_risky shape: {theta_risky.shape}")
          
        #print(f"volatility_buffer shape: {self.volatility_buffer.shape}")     # → (1, n_assets)
        # now incorporate price
        # latest_price: (batch,n_assets)
        # bias = (theta_risky * vol_bias * latest_price)*0                   # → (batch, n_assets)
        # theta_risky = vol_bias * latest_price
        # modulated = (z_risky + bias)*0 + theta_risky
        # return T.softmax(modulated, dim=1)  # Output risky weights
    
        #Use this if wewnated the original version
        bias = (theta_risky * vol_bias * latest_price)                   # → (batch, n_assets)
        modulated = z_risky + bias
        return T.softmax(modulated, dim=1)  # Output risky weights
	
    def save_checkpoint(self):
        print('..saving checkpoint..')
        # ensure directory still exists (in case it was removed)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        print('..loading checkpoint..')
        self.load_state_dict(T.load(self.checkpoint_file))
