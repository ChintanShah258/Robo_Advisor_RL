data_loader.py

    Supporting module only: defines TimeSeriesDataset, a PyTorch Dataset that, given a DataFrame with Dates, raw prices, volumes, rolling vols, embeddings, and reward/meta columns plus a hist_window, yields windowed samples (prices_raw, prices, volume, vol_10d, vol_30d, vol_90d, embeds) along with all of your metadata (mask_meta, date parts, and cumulative targets).

    You don’t invoke this script directly—TimeSeriesDataset will be imported by your RL training code to load the final Excel sheets produced in the next step.

DataProcessor.py – run this to merge your raw market data and embeddings, split into train/validation/test by arbitrary date ranges, compute all of your reward‐target columns and calendar‐based metadata, and export to a multi-sheet Excel file.

    python DataProcessor.py \
      --raw_file "/home/student/robo_advisor_new/base_td3/data_prep/SP500_2015_2024_raw_log_sd.csv" \
      --embed_file "/home/student/robo_advisor_new/extract_embeddings/embedding_results/embedding_sp500_final.csv" \
      --annual_target 0.03 \
      --asset_list FCX DD DIS TJX COP AXP AMGN \
      --hist_window 15 \
      --split \
        'train:2015-01-01|2018-12-31' \
        'train:2021-01-01|2022-12-31' \
        'test:2023-01-01|2024-12-31' \
        'validation:2019-01-01|2020-12-31'

    This will:

        Read your raw CSV (with _log_sd features) and the embeddings CSV (from extract_embeddings.py) by matching on “Dates.”

        Split the merged DataFrame into named subsets (train, test, validation) according to the --split date ranges (you can supply multiple non-contiguous ranges per split).

        For each split, zero-pad the first hist_window–1 rows as “warm-up,” then for the remainder compute:

            Year/month/week features (year_rank, week_of_year, etc.)

            Cumulative and per-period reward targets (daily/weekly/monthly/annual, plus their cumulative sums)

            A binary mask_meta flag (0 for warm-up rows, 1 thereafter)

        Write out an Excel file (auto-named from your last four asset_list symbols) with one sheet per split, each containing:
        mask_meta, Dates, raw & log-sd features, all computed reward/meta columns, and embed_cols.

—
How this ties into the pipeline:

    data_transform.py → prepares your scaled market‐feature CSVs.

    tf_pretrain.py → pre-trains MASTER on 2000–2014, saving best checkpoint.

    extract_embeddings.py → uses that checkpoint to embed 2015–2024 data.

    DataProcessor.py → merges raw + embeddings, splits by date, computes targets & meta → produces the final Excel.

    RL training (not shown here) then uses data_loader.py’s TimeSeriesDataset to load each sheet of that Excel and feed windowed state‐action samples into your agent.