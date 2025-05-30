data_transform.py
Run first to prepare your CSVs and scaler.
python data_transform.py
This reads /home/student/robo_advisor_new/transformer_training/data_sp_500/SP500_Final.csv, splits into 2000–2014 vs. 2015–2024, computes log-returns, log-volumes, and 10/30/90-day rolling volatilities, fits a StandardScaler on 2000–2014 (saving pretrain_scaler.pkl), and writes:

    SP500_Pretrain_Transformer.csv (2000–2014, scaled + Dates)

    /home/student/robo_advisor_new/base_td3/data_prep/SP500_2015_2024.csv (raw 2015–2024)

    /home/student/robo_advisor_new/transformer_training/data_sp_500/SP500_Transformer_Embeddings.csv (2015–2024 transformed)

    plus log–std variants in /base_td3/data_prep/.

tf_pretrain.py
Next, launch your transformer pre-training run.

python tf_pretrain.py \
  --train_data_path /home/student/robo_advisor_new/transformer_training/data_sp_500/SP500_Pretrain_Transformer.csv \
  --n_epochs 20 --lr 1e-5 --window_size 50 --num_of_batches 1 \
  --overlapping False --shuffle_windows True \
  --input_feature_ranges 1-3,10-20 --gate_input_ranges 1-3 \
  --d_model 128 --t_nhead 4 --s_nhead 2 \
  --T_dropout_rate 0.5 --S_dropout_rate 0.5 \
  --beta 0.75 --lambda_contrastive 0.1 --temperature 0.1 \
  --save_prefix sp500_master --save_path model/

This initializes a W&B run, splits data 80/20, builds DataLoaders via WindowSampler (in data_utils.py), constructs your MASTER transformer + projection head (in transformer.py), trains for 20 epochs (MSE + contrastive loss), logs metrics to WandB, and saves the best checkpoint under model/.

sweep_transformer.txt
If you prefer to run a hyperparameter sweep instead of fixed args:

    wandb sweep sweep_transformer.yml  
    wandb agent chintanshah4-carleton-university/robo-advisor_transformer_training/6nj6s2tt --count 18

    This will launch 18 agents to explore your sweep_transformer.yml. When the sweep finishes, go on to step 4.

    aggregate_sweep.py
    Once you’ve run either the fixed-param training or a sweep, aggregate all your WandB runs:
    python aggregate_sweep.py
    This connects to WandB (using the hard-coded SWEEP_URI), fetches every run’s config & summary, writes all_sweep_results.xlsx with a “summary” sheet plus one sheet per run’s history, prints the best run ID and its validation loss, and downloads that run’s best checkpoint into best_sweep_checkpoint/.

    pre_trained_model_chk.py
    Finally, inspect any saved checkpoint:
    python pre_trained_model_chk.py
    It loads robo_advisor/transformer_training/model/sp500_master_pretrained.pth (or your best run’s best .pth) and prints the parameter names and shapes for both the transformer trunk (model) and the reconstruction head (input_proj).

Supporting modules you don’t invoke directly but which are used under the hood:

    data_utils.py defines the WindowSampler dataset.

    tf_pretraining_setup.py contains the SequenceModel trainer class.

    transformer.py houses all transformer components and the MASTER model.


