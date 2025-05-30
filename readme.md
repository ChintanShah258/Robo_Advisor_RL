cat > README.md << 'EOF'
This repository implements a full pipeline for market-data processing, Transformer pretraining, embedding extraction, data merging, and TD3 reinforcement-learning for a robo-advisor. The top-level layout is:

.
├── transformer_training/
├── extract_embeddings/
├── embeddings_merged_data/
└── base_td3/

1. transformer_training/

Contains everything needed to preprocess S&P 500 data and pretrain the MASTER Transformer.

transformer_training/
├── data_sp_500/                    # Raw & processed CSVs
│   ├── SP500_Final.csv             # Original raw data
│   ├── SP500_Pretrain_Transformer.csv  # 2000–2014 scaled + Dates
│   ├── SP500_Transformer_Embeddings.csv # 2015–2024 scaled + Dates
│   └── …                          
├── pretrain_scaler.pkl             # StandardScaler fit on 2000–2014
├── model/                          # Saved checkpoints (post-pretraining)
├── best_sweep_checkpoint/          # Downloaded best‐run checkpoint from W&B
├── all_sweep_results.xlsx          # Aggregated sweep metrics
├── sweep_transformer.yml           # W&B sweep configuration
├── aggregate_sweep.py              # Fetch & aggregate W&B sweep runs  
├── pre_trained_model_chk.py        # Inspect saved .pth parameter shapes  
├── data_transform.py               # CSV prep + scaler fitting  
├── data_utils.py                   # `WindowSampler` dataset for pretraining  
├── tf_pretrain.py                  # Launch MASTER pretraining run  
├── tf_pretraining_setup.py         # `SequenceModel` trainer class  
└── transformer.py                  # PositionalEncoding, Gates, T/SAttention, MASTER  

    data_transform.py

        Reads SP500_Final.csv.

        Splits into 2000–2014 vs 2015–2024.

        Computes log-returns, log-volumes, and 10/30/90-day volatilities.

        Fits StandardScaler on 2000–2014 → pretrain_scaler.pkl.

        Writes out scaled CSVs under data_sp_500/.

    tf_pretrain.py

        Uses data_utils.WindowSampler to build train/val windows.

        Instantiates MASTER from transformer.py and projection head.

        Trains for N epochs (MSE + InfoNCE contrastive loss) via SequenceModel.

        Logs to WandB, saves best checkpoint under model/.

    Hyperparameter Sweep

        sweep_transformer.yml defines the search space.

        Launch with:

        wandb sweep transformer_training/sweep_transformer.yml
        wandb agent YOUR_PROJECT/SWEEP_ID --count 18

    Aggregation & Inspection

        aggregate_sweep.py pulls down all sweep runs, writes all_sweep_results.xlsx, downloads best checkpoint to best_sweep_checkpoint/.

        pre_trained_model_chk.py loads a .pth and prints out the names & shapes of every tensor.

2. extract_embeddings/

Slides a fixed window over unseen data and produces MASTER embeddings.

extract_embeddings/
├── extract_embeddings.py           # Embedding extraction script  
├── embedding_results/              # Output CSVs of windowed embeddings  
├── extract_embeddings_readme       # (text) usage notes  
└── embedding_extraction.txt        # (text) example command  

    extract_embeddings.py

        Reads a CSV with Dates + features.

        Loads pretrained MASTER (aggregate_output=True).

        Slides a window of length T → single 1×d_model embedding each.

        Associates embedding with the date at window end.

        Writes {save_prefix}_sp500_final.csv under embedding_results/.

    Usage example (in embedding_extraction.txt):

    python extract_embeddings.py \
      --unseen_data_path transformer_training/data_sp_500/SP500_Transformer_Embeddings.csv \
      --model_path best_sweep_checkpoint/model/sp500_master_<ID>_best.pth \
      --output_dir embedding_results \
      --save_prefix embedding \
      --d_model 128 --t_nhead 4 --s_nhead 2 \
      --T_dropout_rate 0.1 --S_dropout_rate 0.1 \
      --input_feature_ranges 0-200 \
      --gate_input_ranges 200-500 \
      --beta 0.75 --window_size 15 \
      --aggregate_output True

3. embeddings_merged_data/

Holds the final merged Excel sheets that combine raw market data with embeddings and compute RL reward targets.

embeddings_merged_data/
└── <YOUR_ASSETS>_final_input_data.xlsx  # One sheet per split: train/validation/test  

These files are produced by DataProcessor.py (see next section) and contain columns in this order:

mask_meta, Dates, [raw & log_sd features],
[calendar & reward target columns], [embedding columns]

4. base_td3/

All code for TD3 training, environment, replay buffer, logging, and hyperparameter sweeps.

base_td3/
├── __init__.py
├── ActorNetwork.py            # TD3 actor network  
├── CriticNetwork.py           # TD3 critic network  
├── ReplayBuffer.py            # Batch-aware circular buffer  
├── PortfolioEnv.py            # Custom gym.Env for portfolio sim  
├── logging_utils.py           # LogManager → CSV + Excel  
├── hyper_config.py            # Grid lists for sweep_runner  
├── main.py                    # Single-run training & evaluation  
├── sweep_final.py             # W&B sweep agent entrypoint  
├── sweep_final.yml            # (duplicate) W&B sweep config  
├── sweep_runner.py            # Pure-Python grid search  
├── base_td3_readme            # Overview of base_td3/ contents  
└── data_prep/                 # (optional) supporting data files  

Core components

    ActorNetwork.py

        MLP+LayerNorm actor mapping flattened state →

            w_base (conservative weights via softmax)

            w_risky (logits modulated by θ + volatility bias → softmax)

            amount_risky (λ via sigmoid)

            theta (risk modulation via sigmoid)

    CriticNetwork.py

        Q-network: concatenates state & action → two LayerNorm FC layers → scalar Q

    ReplayBuffer.py

        Circular buffer that handles batches of transitions and samples minibatches.

    PortfolioEnv.py

        Packs price/vol/embedding histories + PV/action memory + pacing features → flat obs

        Decodes action into portfolio allocations → returns, PV update, Sharpe/penalties, dual-ascent on calendar rollovers

    logging_utils.py

        LogManager streams per-step records to CSV (train/validation/test), then finalize() bundles them into one multi-sheet Excel.

Training entrypoints

    Single run

python base_td3/main.py \
  --asset_list ECL NEM APD \
  [--resume_training | --redo_training] \
  --update_every 40 \
  --gradient_steps 20 \
  --T_max 864 \
  --eta_min 1e-6

    Loads embeddings_merged_data/<…>.xlsx via TimeSeriesDataset (in main.py).

    Builds Env & Agent, configures W&B & LogManager.

    Runs TD3 training loop with periodic validation, checkpointing (latest, best_val), final test, and Excel report.

Weights & Biases sweep (sweep_final.py)

    Wraps the same logic as main.py in a train() function that reads wandb.config.

    Invoke with:

        wandb sweep base_td3/sweep_final.yml
        wandb agent YOUR_PROJECT/SWEEP_ID

    Grid search runner (sweep_runner.py)

        Iterates over Cartesian product in hyper_config.py.

        For each combo, spins up a W&B run, trains & validates exactly as main.py, bundles an Excel + hyperparams sheet, then finishes.

🏁 End-to-End Workflow

    Preprocess & scale

cd transformer_training
python data_transform.py

Pretrain MASTER

python tf_pretrain.py [args…]
# or launch W&B sweep

Aggregate sweep & inspect

python aggregate_sweep.py
python pre_trained_model_chk.py

Extract embeddings

cd ../extract_embeddings
python extract_embeddings.py [args…]

Merge + compute RL targets

# run DataProcessor.py in embeddings_merged_data folder
python ../DataProcessor.py [args…]

Train TD3 agent

cd ../base_td3
run W&B sweep: sweep_final.py
# or python main.py [args…]
# or grid search: sweep_runner.py
