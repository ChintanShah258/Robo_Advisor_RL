import os
import sys
import pandas as pd
import numpy as np
import torch as T
from datetime import datetime
import time
import argparse
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from data_prep.data_loader import TimeSeriesDataset
from PortfolioEnv import PortfolioEnv
#from PortfolioEnv_copy import PortfolioEnv_copy
from AgentNetwork import Agent
from logging_utils import LogManager

import wandb


DATA_PATH = '/home/student/robo_advisor_new/base_td3/data_prep/TJX_COP_AXP_AMGN_final_input_data.xlsx'
# ------------ Change this to New Input File when you have new Input data file! ------------
# ANy changes made above have to be made in the DATA_PATH parametr in the file
# hyper_config.py as well
#EXCEL_OUTPUT_DIR = 'base_td3/logs'  # where to put final .xlsx

def collate_str_dates(batch):
    for d in batch:
        d['arr_date'] = str(d['arr_date'])
    return default_collate(batch)


def make_loader(sheet_name, hist_window, n_embeddings, asset_list, batch_size=1):
    df = pd.read_excel(DATA_PATH, sheet_name=sheet_name)
    ds = TimeSeriesDataset(df, hist_window=hist_window, n_embeddings=n_embeddings, asset_list=asset_list)
    return DataLoader(ds,
                      batch_size=batch_size,
                      shuffle=False,
                      drop_last=False,
                      collate_fn=collate_str_dates)


def write_combined_excel(log_dir: str, out_file: str, config: dict):
    """
    1) Write a 'config' sheet
    2) Append train/validation/test sheets from CSVs
    3) Delete the source CSVs once done
    """
    # 1) config
    df_cfg = pd.DataFrame(list(config.items()), columns=['parameter', 'value'])
    with pd.ExcelWriter(out_file) as writer:
        df_cfg.to_excel(writer, sheet_name='config', index=False)
        # 2) data phases
        for phase in ('train','validation','test'):
            csv_path = os.path.join(log_dir, f"{phase}_logs.csv")
            pd.read_csv(csv_path).to_excel(writer, sheet_name=phase, index=False)
    print(f"✅  Combined Excel written to {out_file}")

    # 3) cleanup
    for phase in ('train','validation','test'):
        try:
            os.remove(os.path.join(log_dir, f"{phase}_logs.csv"))
        except OSError:
            pass


def evaluate(agent, env, loader, logger, phase="validation", episode=None):
    env.set_data_loader(loader)
    agent.actor.eval()
    all_sharpes, all_rewards = [], []
    all_month_rets, all_year_rets = [], []
    with T.no_grad():
        state, done = env.reset(), False
        while not done:
            action = agent.choose_action(state)
            logger.log_before(phase, env, action, episode=episode)

            next_state, reward, done, info = env.step(action)
            logger.log_after(phase, env, reward, info, episode=episode)

            all_sharpes.extend(env.sharpe.tolist())
            all_rewards.extend(np.array(reward).tolist())
            all_month_rets.extend(env.r_month.tolist())
            all_year_rets.extend(env.r_year.tolist())
            state = next_state

    agent.actor.train()
    # compute average monthly sharpe from env.monthly_sharpes
    monthly = env.monthly_sharpes[0]
    avg_monthly = float(np.mean(monthly)) if len(monthly)>0 else 0.0
    yearly = env.yearly_sharpes[0]
    avg_yearly = float(np.mean(yearly)) if len(yearly)>0 else 0.0
    avg_month_ret = float(np.mean(all_month_rets)) if all_month_rets else 0.0
    avg_year_ret  = float(np.mean(all_year_rets))  if all_year_rets  else 0.0
    return np.mean(all_sharpes), np.sum(all_rewards), avg_monthly, avg_yearly, avg_month_ret, avg_year_ret

def train_one_episode(agent, env, loader, logger, ep, update_every=40, gradient_steps=20):
    """
    Run one full episode of training:
      - steps until data is exhausted
      - logs before/after to logger
      - accumulates rewards, Sharpe, and losses

    Returns:
      total_reward (float),
      mean_sharpe (float),
      avg_monthly_sharpe (float),
      avg_critic_loss (float),
      avg_actor_loss (float),
      ep_time (float seconds)
    """
    # ── reset env & start timer ─────────────────────
    env.set_data_loader(loader)
    state, done = env.reset(), False
    step_count = 0
    ep_start = time.perf_counter()

    # ── storage for metrics ─────────────────────────
    ep_rewards, ep_sharpes = [], []
    ep_month_rets, ep_year_rets = [], []
    critic_losses, actor_losses = [], []

    # ── interact until data exhausted ───────────────
    while not done:
        action = agent.choose_action(state)
        logger.log_before('train', env, action, episode=ep)

        next_state, reward, done, info = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        logger.log_after('train', env, reward, info, episode=ep)

        ep_rewards.extend(reward.tolist())
        ep_sharpes.extend(env.sharpe.tolist())
        ep_month_rets.extend(env.r_month.tolist())   # <<< NEW
        ep_year_rets.extend(env.r_year.tolist())     # <<< NEW
        state = next_state

        step_count += 1
        # only start learning once warm-up is over
        if agent.time_step >= agent.warm_up and step_count % update_every == 0:
            for _ in range(gradient_steps):
                c_loss, a_loss = agent.learn()
                if c_loss is not None: critic_losses.append(c_loss)
                if a_loss is not None: actor_losses.append(a_loss)

    # ── compute per-episode aggregates ───────────────
    ep_time         = time.perf_counter() - ep_start
    total_reward    = float(np.sum(ep_rewards))
    mean_sharpe     = float(np.mean(ep_sharpes)) if ep_sharpes else 0.0
    avg_critic_loss = float(np.mean(critic_losses)) if critic_losses else 0.0
    avg_actor_loss  = float(np.mean(actor_losses))  if actor_losses  else 0.0

    # ── monthly & yearly sharpe lists from env ──────
    monthly = env.monthly_sharpes[0]
    avg_monthly_sharpe = float(np.mean(monthly)) if len(monthly)>0 else 0.0

    yearly = env.yearly_sharpes[0]
    avg_yearly_sharpe = float(np.mean(yearly)) if len(yearly)>0 else 0.0

    # ── average monthly & yearly return over the episode ──
    avg_monthly_ret = float(np.mean(ep_month_rets)) if ep_month_rets else 0.0
    avg_yearly_ret  = float(np.mean(ep_year_rets))  if ep_year_rets  else 0.0
    
    # ── debug print ─────────────────────────────────
    print(
        f"[DEBUG] Ep {ep}: "
        f"time={ep_time:.2f}s, "
        f"reward={total_reward:.4f}, "
        f"mean_sharpe={mean_sharpe:.4f}, "
        f"avg_monthly_sharpe={avg_monthly_sharpe:.4f}, "
        f"avg_yearly_sharpe={avg_yearly_sharpe:.4f}, "
        f"avg_critic_loss={avg_critic_loss:.6f}, "
        f"avg_actor_loss={avg_actor_loss:.6f}"
    )

    return total_reward, mean_sharpe, avg_monthly_sharpe, avg_yearly_sharpe, \
           avg_monthly_ret, avg_yearly_ret, avg_critic_loss, avg_actor_loss, ep_time

def main(asset_list, resume_training: bool = False, redo_training: bool = False, update_every=40, gradient_steps=20):
    # ─── Hyperparameters ─────────────────────────
    hist_window        = 15 #Always has to be >= 1 or else it will give errors
    # best to match it to the hist_window in the DataProcessor. Will have to make a main config file
    # which controls these values at a global level.
    pv_hist_window     = 15
    action_hist_window = 15
    n_embeddings       = 128
    validation_every   = 5
    num_episodes       = 25
    update_every      = 30
    gradient_steps     = 50

    alpha   = 1e-4
    beta    = 1e-3
    gamma   = 0.99
    tau     = 0.005
    noise   = 0.1
    warm_up = 120

    sharpe_scaling = 0
    sharpe_scaling_monthly = 0
    sharpe_scaling_yearly  = 0
    variance_scaling = 0.1
    monthly_variance_scaling = 0.2
    yearly_variance_scaling  = 0.1
    theta_max      = 3.0
    initial_pv     = 1000
    T_max = 864
    eta_min = 1e-6
    da_ss = 0.2
    
     # ─── NEW: obs‐block switches ───────────────────
    use_volume  = True
    use_embeds = True
    use_vol_10d  = False
    use_vol_30d  = False
    use_vol_90d  = False

    if resume_training and redo_training:
        raise ValueError("Cannot use both --resume_training and --redo_training")

    # ─── Prepare run‐specific folder & config ─────
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    # folder for raw CSV logs
    log_dir = os.path.join("logs", run_id)
    os.makedirs(log_dir, exist_ok=True)
    # folder for final Excel reports
    reports_dir = os.path.join("reports", run_id)
    os.makedirs(reports_dir, exist_ok=True)
    
    #asset_list = ["ECL","NEM","APD"]
    # Don't need it here as we now parse it in the args. 
    # Change this to the list of assets as per the input file
    
    # ─── DataLoaders ──────────────────────────────
    train_loader = make_loader('train',      hist_window, n_embeddings, asset_list)
    val_loader   = make_loader('validation', hist_window, n_embeddings, asset_list)
    test_loader  = make_loader('test',       hist_window, n_embeddings, asset_list)

    # ─── Env & Agent ──────────────────────────────
    env = PortfolioEnv(
        hist_window=hist_window,
        pv_hist_window=pv_hist_window,
        action_hist_window=action_hist_window,
        sharpe_scaling=sharpe_scaling,
        sharpe_scaling_monthly=sharpe_scaling_monthly,
        sharpe_scaling_yearly=sharpe_scaling_yearly,
        variance_scaling=variance_scaling,
        monthly_variance_scaling=monthly_variance_scaling,
        yearly_variance_scaling=yearly_variance_scaling,
        theta_max=theta_max,
        initial_pv=initial_pv,
        use_volume = use_volume,
        use_embeds = use_embeds,
        use_vol_10d = use_vol_10d,
        use_vol_30d = use_vol_30d,
        use_vol_90d = use_vol_90d,
        da_ss=da_ss,
    )
    env.set_data_loader(train_loader)

     # ─── build volatility‐bias vector c from env.vol_10d, vol_30d, vol_90d ─────
    # env.vol_10d etc. are shape (B, n_assets).  We average over batch to get one vector:
    vol10 = env.vol_10d[:, -1, :].squeeze(0)
    vol30 = env.vol_30d[:, -1, :].squeeze(0)
    vol90 = env.vol_90d[:, -1, :].squeeze(0)    # combine with your chosen weights (sum to 1)
    w_vol_10d, w_vol_30d, w_vol_90d = 0.5, 0.3, 0.2
    vol_comp = w_vol_10d*vol10 + w_vol_30d*vol30 + w_vol_90d*vol90
    # normalize if you like
    c = vol_comp / (vol_comp.sum() + 1e-8)
    
    obs_dim  = env.observation_space.shape[0]
    act_dim  = env.action_space.shape[0]
    n_assets = env.prices.shape[-1]
    
    # ─── Checkpointing ───────────────────────────
    config = {
        "run_id": run_id,
        "hist_window": hist_window,
        "pv_hist_window": pv_hist_window,
        "action_hist_window": action_hist_window,
        "n_embeddings": n_embeddings,
        "validation_every": validation_every,
        "num_episodes": num_episodes,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "tau": tau,
        "noise": noise,
        "warm_up": warm_up,
        "theta_max": theta_max,
        "initial_pv": initial_pv,
        "use_volume": use_volume,
        "use_embeds": use_embeds,
        "use_vol_10d": use_vol_10d,
        "use_vol_30d": use_vol_30d,
        "use_vol_90d": use_vol_90d,
        "w_vol_10d":  w_vol_10d,
        "w_vol_30d":  w_vol_30d,
        "w_vol_90d":  w_vol_90d,
        "T_max": T_max,
        "eta_min": eta_min,
        "da_ss": da_ss,
        "sharpe_scaling": sharpe_scaling,
        "sharpe_scaling_monthly": sharpe_scaling_monthly,
        "sharpe_scaling_yearly":  sharpe_scaling_yearly,
        "variance_scaling": variance_scaling,
        "monthly_variance_scaling": monthly_variance_scaling,
        "yearly_variance_scaling":  yearly_variance_scaling,
        }
    
    # ─── WandB ────────────────────────────────────
    wandb.init(
    project="robo-advisor",                # choose or create a project name on wandb.ai
    name=run_id,                           # this will be the run’s name in the UI
    config=config,                         # logs all your hyperparameters automatically
    save_code=True                         # snapshot your code for reproducibility
    )
    
    # ——— define our custom x-axis so episode can go back to 1 each new run ——
    wandb.define_metric("train/episode")
    wandb.define_metric("train/reward",             step_metric="train/episode")
    wandb.define_metric("train/sharpe",             step_metric="train/episode")
    wandb.define_metric("train/avg_monthly_sharpe", step_metric="train/episode")
    wandb.define_metric("train/avg_yearly_sharpe",  step_metric="train/episode")
    wandb.define_metric("validation/episode")
    wandb.define_metric("validation/reward",             step_metric="validation/episode")
    wandb.define_metric("validation/sharpe",             step_metric="validation/episode")
    wandb.define_metric("validation/avg_monthly_sharpe", step_metric="validation/episode")
    wandb.define_metric("validation/avg_yearly_sharpe",  step_metric="validation/episode")
    wandb.define_metric("test/episode")
    wandb.define_metric("test/reward",             step_metric="test/episode")
    wandb.define_metric("test/sharpe",             step_metric="test/episode")
    wandb.define_metric("test/avg_monthly_sharpe", step_metric="test/episode")
    wandb.define_metric("test/avg_yearly_sharpe",  step_metric="test/episode")
    
    # ─── Logger ───────────────────────────────────
    logger = LogManager(log_dir=log_dir, flush_every=2000, write_every=4000)

    agent = Agent(
        alpha=alpha, beta=beta,
        input_dims=obs_dim, tau=tau,
        env=env, gamma=gamma,
        update_actor_interval=5, warm_up=warm_up,
        n_actions=act_dim, n_assets=n_assets,
        max_size=250000, layer1_size=400,
        layer2_size=300, batch_size=40,
        noise=noise,
        vol_bias=c,
        T_max=864,        # from your sweep or config
        eta_min=1e-6     # from your sweep or config
    )

    wandb.watch(agent.actor, log="all", log_freq=100)
    wandb.watch(agent.critic_1, log="all", log_freq=100)
    wandb.watch(agent.critic_2, log="all", log_freq=100)

    best_val_sharpe = -np.inf
    best_val_monthly = -np.inf

    # ─── Optionally resume/redo ───────────────────
    if resume_training or redo_training:
        agent.load_models(label="latest")
        print(f"[Main] Loaded latest checkpoint")
        start_ep = agent.current_episode + 1 if resume_training else 1
    else:
        start_ep = 1

    # ─── Training + Validation Loop ───────────────
    train_phase_start = time.perf_counter()
    for ep in range(start_ep, num_episodes + 1):
        ep_start = time.perf_counter()
        tr_reward, tr_sharpe, tr_monthly_sharpe, tr_yearly_sharpe, \
        tr_monthly_ret, tr_yearly_ret, tr_c_loss, tr_a_loss, tr_time = train_one_episode(
        agent, env, train_loader, logger, ep,update_every=update_every, gradient_steps=gradient_steps)

        ep_time = time.perf_counter() - ep_start
        print(f"[Train] Ep {ep:>2} — Reward {tr_reward:.4f}, Sharpe {tr_sharpe:.4f}  (took {ep_time:.2f}s)")

        # ── Log to W&B ──
        wandb.log({
            "train/episode":           ep,
            "train/reward":            tr_reward,
            "train/sharpe":            tr_sharpe,
            "train/avg_monthly_sharpe": tr_monthly_sharpe,
            "train/avg_yearly_sharpe":  tr_yearly_sharpe,
            "train/avg_monthly_return": tr_monthly_ret,
            "train/avg_yearly_return":  tr_yearly_ret,
            "train/critic_loss":       tr_c_loss,
            "train/actor_loss":        tr_a_loss,
            "train/episode_time_s":    tr_time,
        }, commit=False)
    
        if ep % validation_every == 0:
            val_start = time.perf_counter()
            agent.save_models(label="latest", current_episode=ep)
            print(f"[Main] Saved rolling ’latest’ at ep {ep}")
            vs, vr, vm, vy, vm_ret, vy_ret = evaluate(agent, env, val_loader, logger, phase="validation", episode=ep)
            val_time = time.perf_counter() - val_start
            print(f"[Val]   Ep {ep:>2} — Reward {vr:.4f}, Sharpe {vs:.4f},  (took {val_time:.2f}s)")
            
            wandb.log({
                "validation/episode":         ep,
                "validation/reward":          vr,
                "validation/sharpe":          vs,
                "validation/avg_monthly_sharpe": vm,
                "validation/avg_yearly_sharpe":  vy,
                "validation/avg_monthly_return": vm_ret,
                "validation/avg_yearly_return":  vy_ret,
                "validation/time_s":          val_time,
            }, commit=False)
            
            if vm > best_val_monthly:
                best_val_monthly = vm
                agent.save_models(label="best_val", current_episode=ep)
                print(f"[Main] ▶ New best_val at ep {ep}")

    train_phase_time = time.perf_counter() - train_phase_start
    print(f"[Train] Completed {num_episodes} episodes in {train_phase_time:.2f}s")
    
    # (Optionally) log the total training phase time once
    wandb.log({"train/phase_time_s": train_phase_time}, commit=False)
    
    # ─── Final Test & Snapshot ────────────────────
    agent.load_models(label="best_val")
    test_start = time.perf_counter()
    ts, tr, tm, ty,tm_ret, ty_ret = evaluate(agent, env, test_loader, logger, 'test', None)
    test_time = time.perf_counter() - test_start
    print(f"[Test] — Reward {tr:.4f}, Sharpe {ts:.4f}  (took {test_time:.2f}s)")

    wandb.log({
        "test/reward":            tr,
        "test/sharpe":            ts,
        "test/avg_monthly_sharpe": tm,
        "test/avg_yearly_sharpe":  ty,
        "test/avg_monthly_return": tm_ret,
        "test/avg_yearly_return":  ty_ret,
        "test/time_s":             test_time
    }, commit=False) 

    wandb.log({}, step=ep, commit=True)
    
    agent.save_models(label="final", current_episode=num_episodes)
    print("[Main] Saved final checkpoint")

    # ─── Finish logging & bundle into Excel ───────
    out_xlsx = os.path.join(reports_dir, f"all_logs_{run_id}.xlsx")
    logger.finalize(out_xlsx)
    write_combined_excel(log_dir, out_xlsx, config)

    # ─── Finish the W&B run ───────────────────────
    wandb.finish()

if __name__ == "__main__":

    p = argparse.ArgumentParser()
    # … your existing arguments …
    p.add_argument(
        "--asset_list",
        nargs="+",
        required=True,
        help="List of asset symbols, e.g. ECL NEM APD"
    )
    p.add_argument(
        "--resume_training",
        action="store_true",
        help="Load the ‘latest’ checkpoint at the start"
    )
    p.add_argument(
        "--redo_training",
        action="store_true",
        help="Wipe training history and restart from that checkpoint"
    )
    p.add_argument("--update_every",    type=int, required=False, default=40)
    p.add_argument("--gradient_steps",  type=int, required=False, default=20)
    p.add_argument("--T_max",           type=int,   required=True, help="CosineAnnealing T_max")
    p.add_argument("--eta_min",         type=float, required=True, help="CosineAnnealing eta_min")
    
    args = p.parse_args()

    # now call main with the parsed asset_list and flags
    main(
        asset_list=args.asset_list,
        resume_training=args.resume_training,
        redo_training=args.redo_training
    )

# Way to run this file
# # basic run (no checkpointing)
# python main.py \
#   --asset_list ECL NEM APD

# # resume from latest checkpoint
# python main.py \
#   --asset_list ECL NEM APD \
#   --resume_training

# # wipe history and redo the whole training from latest checkpoint
# python main.py \
#   --asset_list ECL NEM APD \
#   --redo_training