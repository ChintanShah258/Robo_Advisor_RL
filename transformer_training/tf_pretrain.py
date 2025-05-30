# ===========================
# 4. Pre-training MASTER on Data (Without Final Prediction Head)
# ===========================
import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformer import MASTER
from tf_pretraining_setup import SequenceModel
import sys
import os
import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_utils import WindowSampler

def parse_ranges(s: str):
    """Turn "0-9,31-35" → [(0,9),(31,35)] (end exclusive)."""
    out = []
    for part in s.split(','):
        lo, hi = part.split('-')
        out.append((int(lo), int(hi)))
    return out

def load_data_from_csv(file_path):
    df = pd.read_csv(file_path)
    if "Dates" in df.columns:
        df = df.drop(columns=["Dates"])
    return torch.tensor(df.values, dtype=torch.float32)

def main(args):
    # 0) wandb
    run = wandb.init(
        project="robo_advisor_transform_pretrain",  # no slashes
        name=args.save_prefix,
        config=vars(args),
        save_code=True,
        reinit=True
    )
    run_id = run.id
    print(f"Started W&B run name={run.name}, id={run.id}")
    
    # 1) device + data
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data       = load_data_from_csv(args.train_data_path)
    n_total    = len(data)
    n_train    = int(0.8 * n_total)
    train_data = data[:n_train]
    val_data   = data[n_train:]

    # 2) windowed loaders
    train_loader = DataLoader(
        WindowSampler(train_data, window_size=args.window_size,
                      overlapping=args.overlapping,
                      shuffle_windows=args.shuffle_windows),
        batch_size=args.num_of_batches
    )
    val_loader = DataLoader(
        WindowSampler(val_data, window_size=args.window_size,
                      overlapping=args.overlapping,
                      shuffle_windows=args.shuffle_windows),
        batch_size=args.num_of_batches
    )

    # 3) parse your new range args + compute d_feat
    input_ranges = parse_ranges(args.input_feature_ranges)
    gate_ranges  = parse_ranges(args.gate_input_ranges)
    d_feat = sum((hi - lo) for lo, hi in input_ranges)

    # 4) init model
    model = MASTER(
        input_feature_ranges=input_ranges,
        gate_input_ranges = gate_ranges,
        d_feat             = d_feat,
        d_model            = args.d_model,
        t_nhead            = args.t_nhead,
        s_nhead            = args.s_nhead,
        T_dropout_rate     = args.T_dropout_rate,
        S_dropout_rate     = args.S_dropout_rate,
        beta               = args.beta,
        aggregate_output   = False,
    )

    # 5) trainer
    trainer = SequenceModel(
        model              = model,
        d_feat             = d_feat,
        d_model            = args.d_model,
        n_epochs           = args.n_epochs,
        lr                 = args.lr,
        device             = device,
        input_feature_ranges = input_ranges,
        lambda_contrastive = args.lambda_contrastive,
        temperature        = args.temperature,
        save_path          = args.save_path,
        save_prefix        = args.save_prefix
    )

    # 6) train + finish
    train_losses, val_losses = trainer.train(train_loader, val_loader)

    # --- 5) log best model info to W&B summary ---
    wandb.run.summary["best_val_loss"]   = trainer.best_val
    wandb.run.summary["best_epoch"]      = trainer.best_epoch
    wandb.run.summary["best_model_path"] = trainer.best_model_path
    # upload the best‑model checkpoint
    wandb.save(trainer.best_model_path)

    # --- 6) save results to Excel ---
    import pandas as pd
    results_df = pd.DataFrame({
        "epoch":      list(range(1, args.n_epochs + 1)),
        "train_loss": train_losses,
        "val_loss":   val_losses
    })
    config_df = pd.DataFrame.from_dict(vars(args), orient="index", columns=["value"])
    best_df   = pd.DataFrame([{
        "best_epoch":      trainer.best_epoch,
        "best_val_loss":   trainer.best_val,
        "best_model_path": trainer.best_model_path
    }])

    xlsx_path = os.path.join(
        args.save_path,
        f"{args.save_prefix}_{run.id}.xlsx"
    )
    with pd.ExcelWriter(xlsx_path) as writer:
        results_df.to_excel(writer, sheet_name="losses", index=False)
        config_df.to_excel(writer, sheet_name="config")
        best_df.to_excel(writer, sheet_name="best_model", index=False)

    print("Wrote Excel report to", xlsx_path)
    wandb.save(xlsx_path)

    wandb.finish()
    print("Done — best/final saved in", args.save_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--train_data_path',        required=True)
    p.add_argument('--n_epochs',    type=int, default=10)
    p.add_argument('--lr',          type=float, default=1e-5)
    p.add_argument('--window_size', type=int,   default=50)
    p.add_argument('--num_of_batches', type=int, default=1)
    p.add_argument('--overlapping', type=bool,  default=False)
    p.add_argument('--shuffle_windows', type=bool, default=False)
    p.add_argument('--aggregate_output', type=bool, default=False)

    # transformer‐side hyperparams
    p.add_argument('--input_feature_ranges', type=str, required=True,
                   help='e.g. "0-9,31-35"')
    p.add_argument('--gate_input_ranges', type=str, required=True,
                   help='e.g. "10-20,25-30"')
    p.add_argument('--d_model',         type=int, default=128)
    p.add_argument('--t_nhead',         type=int, default=4)
    p.add_argument('--s_nhead',         type=int, default=2)
    p.add_argument('--T_dropout_rate',  type=float, default=0.5)
    p.add_argument('--S_dropout_rate',  type=float, default=0.5)
    p.add_argument('--beta',            type=float, default=1.0)

    # contrastive + wandb
    p.add_argument('--lambda_contrastive', type=float, default=0.1)
    p.add_argument('--temperature',       type=float, default=0.1)

    # saving
    p.add_argument('--save_prefix', type=str, default='sp500_master')
    p.add_argument('--save_path',   type=str, default='model/')

    args = p.parse_args()
    main(args)