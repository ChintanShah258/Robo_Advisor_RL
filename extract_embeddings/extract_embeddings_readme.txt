extract_embeddings.py – after you’ve pre-trained and selected a checkpoint, run this to generate window-based embeddings for unseen data:

    python extract_embeddings.py \
      --unseen_data_path /home/student/robo_advisor_new/transformer_training/data_sp_500/SP500_Transformer_Embeddings.csv \
      --model_path /home/student/robo_advisor_new/transformer_training/best_sweep_checkpoint/model/sp500_master_dpoiek8p_best.pth \
      --output_dir /home/student/robo_advisor_new/extract_embeddings/embedding_results \
      --save_prefix embedding \
      --d_model 128 \
      --t_nhead 4 \
      --s_nhead 2 \
      --T_dropout_rate 0.1 \
      --S_dropout_rate 0.1 \
      --input_feature_ranges 0-200 \
      --gate_input_ranges 200-500 \
      --beta 0.75 \
      --window_size 15 \
      --aggregate_output True

    This script:

        Loads your unseen CSV (with a “Dates” column and features)

        Parses input_feature_ranges & gate_input_ranges

        Instantiates MASTER with aggregate_output=True and loads model weights from the given .pth

        Slides a window of length window_size over the data, runs each through MASTER, and collects the resulting 1×d_model embedding per window

        Associates each embedding with the date at the end of its window

        Writes out {save_prefix}_sp500_final.csv under your output_dir, with “Dates” + embedding columns

—
Full end-to-end order:

    data_transform.py → prepare CSVs & scaler

    tf_pretrain.py → pre-train MASTER & save best checkpoint

    (optional) wandb sweep (via sweep_transformer.txt) → hyperparameter exploration

    aggregate_sweep.py → collect & download best sweep checkpoint

    pre_trained_model_chk.py → inspect your saved .pth

    extract_embeddings.py → generate and save windowed embeddings for unseen data