# aggregate_sweep.py
import os
import pandas as pd
import wandb

SWEEP_URI   = "chintanshah4-carleton-university/robo-advisor_transformer_training/6nj6s2tt"
OUTPUT_XLSX = "all_sweep_results.xlsx"

api   = wandb.Api()
sweep = api.sweep(SWEEP_URI)
runs  = sweep.runs

# collect per‐run data
all_runs = []
for run in runs:
    summary = run.summary
    config  = run.config
    all_runs.append({
        "run_id":        run.id,
        "train_loss":    summary.get("train_loss"),
        "val_loss":      summary.get("val_loss"),
        "best_val_loss": summary.get("best_val_loss"),
        "best_epoch":    summary.get("best_epoch"),
        **{f"cfg_{k}": v for k, v in config.items()}
    })

df = pd.DataFrame(all_runs)

# write one sheet per run with its own losses history if you logged that as a history line:
with pd.ExcelWriter(OUTPUT_XLSX) as writer:
    # summary sheet
    df.to_excel(writer, sheet_name="summary", index=False)

    # optional: per-run full history, if you logged each epoch as history
    for run in runs:
        history = run.history(samples=100000)  # pull all history
        history.to_excel(writer, sheet_name=run.id[:31], index=False)

# find best run
best = min(runs, key=lambda r: r.summary.get("best_val_loss", float("inf")))
print(f"Best run is {best.id} with loss {best.summary['best_val_loss']}")

# download its checkpoint artifact
best_model_file = best.summary["best_model_path"]
# if you used wandb.save(), it’ll be stored under run.dir:
local_path = os.path.join(best.dir, os.path.basename(best_model_file))
print("Best model is at:", local_path)

best_model_name = os.path.basename(best.summary["best_model_path"])
os.makedirs("best_sweep_checkpoint", exist_ok=True)

# wandb.Api().run.files() gives you a list of File objects
for f in best.files():
    if os.path.basename(f.name) == best_model_name:
        local_path = f.download(root="best_sweep_checkpoint", replace=True)
        print(f"✅ Downloaded best checkpoint to: {local_path}")
        break
else:
    print(f"⚠️  Couldn’t find `{best_model_name}` among files for run {best.id}")