ActorNetwork.py

    Defines the TD3 actor: a two-layer MLP with LayerNorm that maps a flattened state →

        w_base (conservative portfolio weights via softmax),

        w_risky (raw logits modulated by a learnable “theta” scalar and a fixed volatility-bias buffer → softmax),

        amount_risky (λ ∈ (0,1) via sigmoid),

        theta (risk modulation via sigmoid).

    Exposes forward(state) → concatenated action vector and save_checkpoint/load_checkpoint for the actor alone.

    Usage: imported by AgentNetwork.py to instantiate online & target actor nets.

CriticNetwork.py

    Defines one Q-network: concatenates state & action → two LayerNorm’d FC layers → single-head Q-value.

    Provides its own optimizer plus save_checkpoint/load_checkpoint for standalone checkpointing.

    Usage: two critics (online & target) created in AgentNetwork.py for TD3.

ReplayBuffer.py

    A fixed-size circular buffer storing batches of transitions (state, action, reward, next_state, done).

    store_transition handles batch inputs; sample_buffer(batch_size) returns random minibatches.

    Usage: used by AgentNetwork.py’s Agent.remember and Agent.learn.

PortfolioEnv.py

    A custom Gym-style environment wrapping your TimeSeriesDataset batches.

        Observation: concatenated histories of prices, (optional) volume/volatility/embeds, PV & action history, dual-ascent features.

        Action: a vector of length 2*n_assets + 2 (w_base, w_risky, λ, θ).

        Computes portfolio returns, PV updates, pacing penalties, Sharpe bonuses, and full dual-ascent updates at calendar boundaries.

    Implements standard reset() → initial obs and step(action) → (next_obs, reward, done, info).

    Usage: passed into AgentNetwork.Agent to simulate episodes.

hyper_config.py

    Central grid definitions for sweeping: hist windows, embedding dims, network layer sizes, RL & env hyperparameters, file paths, etc.

    Usage: imported by sweep_runner.py to instantiate full grid search.

logging_utils.py

    LogManager streams per-step records to CSVs during train/validation/test:

        log_before(…) captures pre-step env fields + planned action,

        log_after(…) fills in post-step PV, reward, metrics, flushes to disk in batches,

        finalize(xlsx_path) bundles CSVs into one Excel with sheets per phase.

    Usage: instantiated in main.py (and sweep scripts) to log everything.

main.py

    Data loaders: uses TimeSeriesDataset on the Excel produced by DataProcessor.py to build DataLoaders for train/validation/test.

    Env & Agent setup:

        Builds PortfolioEnv with user-specified feature blocks, then constructs volatility-bias vector c.

        Instantiates Agent (online & target actors/critics, replay buffer, cosine annealing schedulers).

    W&B & logging: initializes a run, watches networks, and sets up LogManager.

    Training loop: for each episode:

        Interact with env, log before/after, store in buffer, TD3 updates every so many steps, roll validation & checkpointing.

        Tracks best validation monthly Sharpe and saves “best_val” and final checkpoints.

    Testing: loads best_val, runs on test split, logs metrics.

    Reporting: calls logger.finalize(...) and writes a combined Excel with config + all phases.

    CLI: --asset_list, --resume_training, --redo_training, --update_every, --gradient_steps, plus scheduler args.

    Usage:

    python main.py \
      --asset_list ECL NEM APD \
      [--resume_training | --redo_training] \
      --update_every 40 \
      --gradient_steps 20 \
      --T_max 864 \
      --eta_min 1e-6

sweep_final.py

    A W&B-style sweep agent entrypoint. Reads sweep_final.yml, unpacks all hyperparameters from wandb.config, and calls almost the same logic as main.py (but within a single-run function train(resume, redo, args)).

    Configures W&B metrics, creates DataLoaders, Env, Agent, and then runs train_one_episode + evaluate, saving checkpoints on the fly (“latest”, “best_val”, “final”).

    Usage:

    wandb sweep sweep_final.yml        # once
    wandb agent <USERNAME>/robo-advisor/SWEEP_ID

sweep_runner.py

    A pure-Python grid search (no W&B). Iterates over the Cartesian product defined in hyper_config.py, and for each combination:

        Builds a unique run_id tag, initializes a W&B run with that config.

        Creates logs folder, LogManager, DataLoaders, Env, Agent (no vol_bias here unless you compute it on the fly).

        Runs the same training-validation loop as in main.py, saving “latest” & “best_val” checkpoints.

        Bundles all CSVs into an Excel report augmented with a “hyperparams” sheet.

        Calls wandb.finish().

    Usage:

        python sweep_runner.py [--resume_training] [--redo_training]

Pipeline Summary & Run Order

    Data Preparation:

        data_transform.py → scaled CSVs

        tf_pretrain.py → pretrain MASTER → best checkpoint

        extract_embeddings.py → embed 2015–2024 → embeddings CSV

        DataProcessor.py → merge raw + embeddings, split, compute rewards → final Excel

    RL Training & Evaluation:

        One‐off run: python main.py ...

        Hyperparameter sweep (W&B): register with sweep_final.py + sweep_final.yml, then wandb agent ...

        Grid search runner: python sweep_runner.py ...

    Key Modules Used During RL:

        TimeSeriesDataset (from data_loader.py) → DataLoader

        PortfolioEnv → Gym loop

        Agent (ActorNetwork + CriticNetwork + ReplayBuffer + schedulers) → TD3 updates

        LogManager → structured logging & final Excel reports

With this structure, you can start with main.py for a single run, or plug into sweep_final.py for extensive hyperparameter exploration—all built on the same core ActorNetwork, CriticNetwork, ReplayBuffer, and PortfolioEnv.