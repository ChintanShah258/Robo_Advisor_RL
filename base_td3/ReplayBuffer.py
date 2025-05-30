import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, state_shape, n_actions):
        """
        Args:
            max_size (int): Maximum number of transitions to store.
            state_shape (tuple): Shape of the state input, e.g., (embedding_dim + 2,)
                                 where the extra dimensions represent V_base and P_acc.
            n_actions (int): Dimension of the action output. 
                             For example, if your action comprises conservative weights (length N),
                             risky weights (length N), lambda, and theta, then n_actions = 2*N + 2.
        """                         
        self.mem_size = max_size  # Maximum buffer size (shouldn't be unbounded)
        self.mem_cntr = 0         # Counter to keep track of stored experiences
        if isinstance(state_shape, int):
            state_shape = (state_shape,)
        else:
            state_shape = tuple(state_shape)
        print(f"[ReplayBuffer] state_shape={state_shape}, n_actions={n_actions}, max_size={max_size}")
        self.state_memory = np.zeros((self.mem_size, *state_shape))
        self.new_state_memory = np.zeros((self.mem_size, *state_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)  # Stores "done" flags

    def store_transition(self, state, action, reward, state_, done):
        """
        Now handles a batch of size B:
        state:  (B, *state_shape)
        action: (B, n_actions)
        reward: (B,)
        state_: (B, *state_shape)
        done:   (B,)
        """
        B = state.shape[0]
        for i in range(B):
            idx = self.mem_cntr % self.mem_size
            self.state_memory[idx]     = state[i]
            self.new_state_memory[idx] = state_[i]
            self.action_memory[idx]    = action[i]
            self.reward_memory[idx]    = reward[i]
            self.terminal_memory[idx]  = done[i]
            self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        """
        Returns a random sample of experiences from the buffer.
        Args:
            batch_size (int): The number of experiences to sample.
        Returns:
            Tuple of (states, actions, rewards, next_states, dones).
        """
        max_mem = min(self.mem_cntr, self.mem_size)
        batch_indices = np.random.choice(max_mem, batch_size, replace=False)
        
        states = self.state_memory[batch_indices]
        actions = self.action_memory[batch_indices]
        rewards = self.reward_memory[batch_indices]
        states_ = self.new_state_memory[batch_indices]
        dones = self.terminal_memory[batch_indices]
        
        return states, actions, rewards, states_, dones
    
    def save_to_dict(self, replay_window_size: int = 5000):
        """
        Serialize only the last `window_size` transitions.
        If window_size is None, serialize everything.
        """
        max_mem = min(self.mem_cntr, self.mem_size)
        if replay_window_size is None or replay_window_size >= max_mem:
            idxs = np.arange(max_mem)
        else:
            # pick the last window_size indexes in circular order
            start = (self.mem_cntr - replay_window_size) % self.mem_size
            if start + replay_window_size <= self.mem_size:
                idxs = np.arange(start, start + replay_window_size)
            else:
                # wrap around
                idxs = np.concatenate([
                    np.arange(start, self.mem_size),
                    np.arange(0, (start + replay_window_size) % self.mem_size)
                ])
        return {
            'state':     self.state_memory[idxs].copy(),
            'new_state': self.new_state_memory[idxs].copy(),
            'action':    self.action_memory[idxs].copy(),
            'reward':    self.reward_memory[idxs].copy(),
            'done':      self.terminal_memory[idxs].copy(),
            'mem_cntr':  self.mem_cntr,
        }
        
    def load_from_dict(self, d: dict):
        arr_len = len(d['reward'])
        # fill arrays from 0 to arr_len
        self.state_memory[:arr_len]     = d['state']
        self.new_state_memory[:arr_len] = d['new_state']
        self.action_memory[:arr_len]    = d['action']
        self.reward_memory[:arr_len]    = d['reward']
        self.terminal_memory[:arr_len]  = d['done']
        self.mem_cntr = d.get('mem_cntr', arr_len)
