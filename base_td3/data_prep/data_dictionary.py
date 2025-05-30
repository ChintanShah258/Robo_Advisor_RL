import pandas as pd

data = [
    ("Initialization", "self.data", "DataFrame", "Full input table (dates, market columns, metadata, embeddings, pacing targets)."),
    ("Initialization", "self.initial_pv", "float", "Start‐of‐year portfolio value (e.g. 1000)."),
    ("Initialization", "self.annual_rate", "float", "Nominal annual growth rate (e.g. 0.03)."),
    ("Initialization", "self.n_embeddings", "int", "Number of embedding columns at end of DataFrame."),
    ("Initialization", "self.lgr_mult_weekly", "float", "Lagrange multiplier for weekly pacing."),
    ("Initialization", "self.lgr_mult_monthly", "float", "Lagrange multiplier for monthly pacing."),
    ("Initialization", "self.lgr_mult_annual", "float", "Lagrange multiplier for annual pacing."),
    ("Initialization", "self.da_ss", "float", "Dual‐ascent step size (η)."),
    ("Initialization", "self.hist_window", "int", "Window length for rolling Sharpe."),
    ("Initialization", "self.sharpe_scaling", "float", "Weight on rolling‐Sharpe bonus."),
    ("Initialization", "self.recent_returns", "deque", "Buffer of last hist_window rewards for Sharpe."),
    ("Initialization", "self.theta_max", "float", "Maximum θ_risky for leverage use‐case."),
    ("Data Arrays", "self.prices", "ndarray (T×n)", "Raw asset prices."),
    ("Data Arrays", "self.embeds", "ndarray (T×m)", "Market embeddings."),
    ("Data Arrays", "self.arr_week", "ndarray (T,)", "Week indices."),
    ("Data Arrays", "self.arr_month", "ndarray (T,)", "Month indices."),
    ("Data Arrays", "self.arr_year", "ndarray (T,)", "Year values."),
    ("Data Arrays", "self.arr_year_rank", "ndarray (T,)", "Sequential year counter."),
    ("Data Arrays", "self.arr_et_w", "ndarray (T,)", "Weekly pacing fractions."),
    ("Data Arrays", "self.arr_et_m", "ndarray (T,)", "Monthly pacing fractions."),
    ("Data Arrays", "self.arr_et_y", "ndarray (T,)", "Annual pacing fractions."),
    ("Data Arrays", "self.expected_start", "dict", "Year→start PV baseline."),
    ("State Trackers", "self.current_step", "int", "Index into time series."),
    ("State Trackers", "self.pv", "float", "Current portfolio value."),
    ("State Trackers", "self.prev_pv", "float", "Previous portfolio value."),
    ("State Trackers", "self.pv_history", "list", "Last hist_window PVs."),
    ("State Trackers", "self.action_history", "list", "Last hist_window actions."),
    ("State Trackers", "self.week_start_pv", "float", "PV at start of current week."),
    ("State Trackers", "self.month_start_pv", "float", "PV at start of current month."),
    ("State Trackers", "self.year_start_pv", "float", "PV at start of current year."),
    ("Step Locals", "action", "ndarray", "Action vector from Actor."),
    ("Step Locals", "w_base", "ndarray", "Conservative weights."),
    ("Step Locals", "w_risk", "ndarray", "Risky sleeve weights."),
    ("Step Locals", "amount_risky", "float", "λ_risky ∈ [0,1]."),
    ("Step Locals", "theta_risky", "float", "θ_risky parameter."),
    ("Step Locals", "weights", "ndarray", "Final portfolio weights."),
    ("Step Locals", "asset_rets", "ndarray", "Daily asset returns."),
    ("Step Locals", "daily_return", "float", "Portfolio daily return."),
    ("Step Locals", "r_week", "float", "Weekly cumulative return fraction."),
    ("Step Locals", "r_month", "float", "Monthly cumulative return fraction."),
    ("Step Locals", "r_year", "float", "Annual cumulative return fraction."),
    ("Step Locals", "et_w", "float", "Daily weekly target fraction."),
    ("Step Locals", "et_m", "float", "Daily monthly target fraction."),
    ("Step Locals", "et_y", "float", "Daily annual target fraction."),
    ("Step Locals", "pen_w", "float", "Weekly penalty term."),
    ("Step Locals", "pen_m", "float", "Monthly penalty term."),
    ("Step Locals", "pen_y", "float", "Annual penalty term."),
    ("Step Locals", "r_env", "float", "Reward before Sharpe bonus."),
    ("Step Locals", "sharpe", "float", "Rolling Sharpe."),
    ("Step Locals", "reward", "float", "Final RL reward."),
    ("Step Locals", "done", "bool", "Episode termination flag.")
]
df = pd.DataFrame(data, columns=["Category","Variable","Type","Description"])

# write to Excel
with pd.ExcelWriter('data_dictionary.xlsx', engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='PortfolioEnv', index=False)
print("Wrote PortfolioEnv dictionary to data_dictionary.xlsx") 

# Define ActorNetwork variable dictionary
actor_data = [
    ("Initialization", "alpha", "float", "Learning rate for Adam optimizer."),
    ("Initialization", "input_dims", "int", "Dimension of state input vector."),
    ("Initialization", "fc1_dims", "int", "Number of units in first hidden layer."),
    ("Initialization", "fc2_dims", "int", "Number of units in second hidden layer."),
    ("Initialization", "n_stocks", "int", "Number of assets (size of weight vectors)."),
    ("Initialization", "name", "str", "Name used for checkpoint files."),
    ("Initialization", "checkpoint_dir", "str", "Directory for saving/loading checkpoints."),
    ("Initialization", "checkpoint_file", "str", "Full path to checkpoint file."),
    ("Initialization", "fc1", "nn.Linear", "First fully connected layer."),
    ("Initialization", "fc2", "nn.Linear", "Second fully connected layer."),
    ("Initialization", "base_head", "nn.Linear", "Head producing base portfolio logits."),
    ("Initialization", "risky_head_raw", "nn.Linear", "Head producing raw risky portfolio logits."),
    ("Initialization", "amount_risky", "nn.Linear", "Head producing scalar lambda_risky logits."),
    ("Initialization", "theta_head", "nn.Linear", "Head producing scalar theta_risky logits."),
    ("Initialization", "optimizer", "optim.Adam", "Adam optimizer for actor parameters."),
    ("Initialization", "device", "torch.device", "Computation device (cpu or cuda)."),
    ("Forward Pass", "state", "Tensor", "Input state tensor of shape (batch_size, input_dims)."),
    ("Forward Pass", "x", "Tensor", "Latent representation after two FC+ReLU layers."),
    ("Forward Pass", "w_base", "Tensor", "Softmax over base_head logits → conservative weights."),
    ("Forward Pass", "theta_risky", "Tensor", "Sigmoid(theta_head) ∈ (0,1) risk‐bias parameter."),
    ("Forward Pass", "z_risky", "Tensor", "Raw risky logits from risky_head_raw."),
    ("Forward Pass", "w_risky", "Tensor", "Softmax(modulate_risky_weights) → risky weights."),
    ("Forward Pass", "amount_risky", "Tensor", "Sigmoid(amount_risky) ∈ (0,1) fraction to risky sleeve."),
    ("Forward Pass", "action", "Tensor", "Concatenation [w_base, w_risky, amount_risky, theta_risky]."),
    ("modulate_risky_weights", "z_risky", "Tensor", "Raw risky logits input."),
    ("modulate_risky_weights", "theta_risky", "Tensor", "Risk bias scalar input."),
    ("modulate_risky_weights", "risk_bias", "Tensor", "Placeholder bias vector (ones)."),
    ("modulate_risky_weights", "modulated", "Tensor", "z_risky + theta_risky * risk_bias."),
    ("modulate_risky_weights", "return", "Tensor", "Softmax(modulated) → final risky weights."),
    ("Checkpoint", "save_checkpoint()", "method", "Saves model state to checkpoint_file."),
    ("Checkpoint", "load_checkpoint()", "method", "Loads model state from checkpoint_file.")
]

actor_df = pd.DataFrame(actor_data, columns=["Category", "Variable", "Type", "Description"])

# Write both dataframes to Excel
with pd.ExcelWriter('data_dictionary.xlsx', engine='openpyxl', mode='a') as writer:
    actor_df.to_excel(writer, sheet_name='ActorNetwork', index=False)
print("Wrote ActorNetwork dictionary to data_dictionary.xlsx")

# Define CriticNetwork variable dictionary
critic_data = [
    ("Initialization", "beta", "float", "Learning rate for Adam optimizer."),
    ("Initialization", "input_dims", "tuple", "Shape of state input (e.g., (obs_dim,))."),
    ("Initialization", "fc1_dims", "int", "Number of units in first hidden layer."),
    ("Initialization", "fc2_dims", "int", "Number of units in second hidden layer."),
    ("Initialization", "n_actions", "int", "Dimension of action input."),
    ("Initialization", "name", "str", "Name used for checkpointing."),
    ("Initialization", "checkpoint_dir", "str", "Directory to save checkpoints."),
    ("Initialization", "checkpoint_file", "str", "Full path to checkpoint file."),
    ("Initialization", "fc1", "nn.Linear", "First fully connected layer: maps [state+action] to fc1_dims."),
    ("Initialization", "fc2", "nn.Linear", "Second fully connected layer: maps fc1_dims to fc2_dims."),
    ("Initialization", "q1", "nn.Linear", "Output layer: maps fc2_dims to single Q-value."),
    ("Initialization", "optimizer", "optim.Adam", "Adam optimizer for critic parameters."),
    ("Initialization", "device", "torch.device", "Computation device (cpu or cuda)."),
    ("Forward Pass", "state", "Tensor", "State input tensor of shape (batch_size, input_dims)."),
    ("Forward Pass", "action", "Tensor", "Action input tensor of shape (batch_size, n_actions)."),
    ("Forward Pass", "q1_action_value", "Tensor", "Latent tensor after fc1 and ReLU."),
    ("Forward Pass", "q1", "Tensor", "Final Q-value tensor of shape (batch_size, 1)."),
    ("Checkpoint", "save_checkpoint()", "method", "Save model state to checkpoint_file."),
    ("Checkpoint", "load_checkpoint()", "method", "Load model state from checkpoint_file.")
]
critic_df = pd.DataFrame(critic_data, columns=["Category", "Variable", "Type", "Description"])

# Write to Excel
with pd.ExcelWriter('data_dictionary.xlsx', engine='openpyxl', mode='a') as writer:
    critic_df.to_excel(writer, sheet_name='CriticNetwork', index=False)
print("Wrote CriticNetwork dictionary to data_dictionary.xlsx")

# Define Agent variable dictionary
agent_data = [
    ("Initialization", "alpha", "float", "Actor learning rate."),
    ("Initialization", "beta", "float", "Critic learning rate."),
    ("Initialization", "input_dims", "tuple", "Shape of state vector."),
    ("Initialization", "tau", "float", "Polyak averaging factor for target networks."),
    ("Initialization", "env", "gym.Env", "The environment instance."),
    ("Initialization", "gamma", "float", "Discount factor."),
    ("Initialization", "update_actor_iter", "int", "How often to update actor."),
    ("Initialization", "warmup", "int", "Number of random steps before policy use."),
    ("Initialization", "n_actions", "int", "Action dimension."),
    ("Initialization", "max_action", "array", "Upper bounds from env.action_space.high."),
    ("Initialization", "min_action", "array", "Lower bounds from env.action_space.low."),
    ("Initialization", "memory", "ReplayBuffer", "Experience replay buffer."),
    ("Initialization", "batch_size", "int", "Minibatch size for learning."),
    ("Initialization", "learn_step_cntr", "int", "Counter for actor updates."),
    ("Initialization", "time_step", "int", "Global step counter."),
    ("Initialization", "actor", "ActorNetwork", "The online actor network."),
    ("Initialization", "critic_1", "CriticNetwork", "First online critic."),
    ("Initialization", "critic_2", "CriticNetwork", "Second online critic."),
    ("Initialization", "target_actor", "ActorNetwork", "Target actor network."),
    ("Initialization", "target_critic_1", "CriticNetwork", "Target critic 1."),
    ("Initialization", "target_critic_2", "CriticNetwork", "Target critic 2."),
    ("Initialization", "noise", "float", "Scale of exploration noise."),
    ("choose_action", "observation", "ndarray", "Current state observation."),
    ("choose_action", "pi", "Tensor", "Action logits or sample."),
    ("choose_action", "pi_prime", "Tensor", "Noisy and clamped action."),
    ("remember", "state, action, reward, new_state, done", "various", "Experience tuple stored in memory."),
    ("learn", "state, action, reward, new_state, done", "various", "Batch sampled from memory."),
    ("learn", "target_actions", "Tensor", "Noisy target actor actions."),
    ("learn", "q1_, q2_", "Tensor", "Target Q-values."),
    ("learn", "q1, q2", "Tensor", "Online Q-values."),
    ("learn", "critic_value_", "Tensor", "Min of q1_ and q2_."),
    ("learn", "target", "Tensor", "Bellman target values."),
    ("learn", "q1_loss, q2_loss", "Tensor", "MSE losses for critics."),
    ("learn", "actor_loss", "Tensor", "Policy gradient loss."),
    ("update_network_parameters", "tau", "float", "Soft update interpolation factor."),
    ("update_network_parameters", "actor_params, critic_params", "dict", "Parameter dicts for interpolation."),
    ("update_network_parameters", "save_models()", "method", "Save all model checkpoints."),
    ("update_network_parameters", "load_models()", "method", "Load all model checkpoints.")
]
agent_df = pd.DataFrame(agent_data, columns=["Category", "Variable", "Type", "Description"])

# Append to existing Excel
with pd.ExcelWriter('data_dictionary.xlsx', engine='openpyxl', mode='a') as writer:
    agent_df.to_excel(writer, sheet_name='Agent', index=False)
print("Wrote AgentNetwork dictionary to data_dictionary.xlsx")

# Define ReplayBuffer variable dictionary
replay_data = [
    ("Initialization", "max_size", "int", "Maximum number of transitions to store."),
    ("Initialization", "state_shape", "tuple", "Shape of each state vector."),
    ("Initialization", "n_actions", "int", "Dimension of the action vector."),
    ("Initialization", "mem_size", "int", "Alias for max_size."),
    ("Initialization", "mem_cntr", "int", "Counter for total stored experiences."),
    ("Initialization", "state_memory", "ndarray", "Array of shape (mem_size, *state_shape) storing states."),
    ("Initialization", "new_state_memory", "ndarray", "Array storing next states."),
    ("Initialization", "action_memory", "ndarray", "Array of shape (mem_size, n_actions) storing actions."),
    ("Initialization", "reward_memory", "ndarray", "Array of shape (mem_size,) storing rewards."),
    ("Initialization", "terminal_memory", "ndarray", "Boolean array of shape (mem_size,) storing done flags."),
    ("store_transition", "state", "array-like", "Current state vector."),
    ("store_transition", "action", "array-like", "Action taken."),
    ("store_transition", "reward", "float", "Reward received."),
    ("store_transition", "state_", "array-like", "Next state vector."),
    ("store_transition", "done", "bool", "Episode termination flag."),
    ("store_transition", "index", "int", "Position in buffer to overwrite."),
    ("store_transition", "mem_cntr", "int", "Incremented after storing."),
    ("sample_buffer", "batch_size", "int", "Number of samples to return."),
    ("sample_buffer", "max_mem", "int", "Effective size of buffer to sample from."),
    ("sample_buffer", "batch_indices", "ndarray", "Random indices selected for sampling."),
    ("sample_buffer", "states", "ndarray", "Batch of states."),
    ("sample_buffer", "actions", "ndarray", "Batch of actions."),
    ("sample_buffer", "rewards", "ndarray", "Batch of rewards."),
    ("sample_buffer", "states_", "ndarray", "Batch of next states."),
    ("sample_buffer", "dones", "ndarray", "Batch of done flags."),
]

replay_df = pd.DataFrame(replay_data, columns=["Category", "Variable", "Type", "Description"])

# Append to existing Excel
with pd.ExcelWriter('data_dictionary.xlsx', engine='openpyxl', mode='a') as writer:
    replay_df.to_excel(writer, sheet_name='ReplayBuffer', index=False)
print("Wrote ReplayBuffer dictionary to data_dictionary.xlsx")