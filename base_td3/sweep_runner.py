#!/usr/bin/env python3
"""
This script runs a full hyperparameter sweep over combinations defined in hyper_config.py.

Usage:
  python sweep_runner.py

Outputs per run:
  - CSV logs (train, validation, test) under logs/<run_id>/
  - An Excel file reports/all_logs_<run_id>_<timestamp>.xlsx with sheets: train, validation, test, hyperparams
"""

import os
import itertools
import pandas as pd
import time
import wandb
import argparse
from datetime import datetime

# import base_td3.AgentNetwork as _AN
# print(f"[IMPORT_DEBUG] AgentNetwork loaded from: {_AN.__file__}")

from main import (
    train_one_episode,
    evaluate,
    make_loader,
    PortfolioEnv,
    Agent,
    LogManager
)
from hyper_config import (
    HIST_WINDOWS,
    PV_HIST_WINDOWS,
    ACTION_HIST_WINDOWS,
    EMBEDDING_DIMS,
    NUM_EPISODES,
    VALIDATION_EVERY,
    RESUME_EVERY,
    ALPHA,
    BETA,
    GAMMA,
    TAU,
    NOISE_SCALE,
    WARM_UP,
    SHARPE_SCALING,
    SHARPE_SCALING_MONTHLY,
    SHARPE_SCALING_YEARLY,
    THETA_MAX,
    INITIAL_PV,
    LABEL_LATEST,
    LABEL_BEST_VAL,
    LABEL_FINAL,
    DATA_PATH,
    LAYER1_SIZE,
    LAYER2_SIZE,
    ASSET_LIST,
    W_VOL_10D,
    W_VOL_30D,
    W_VOL_90D,
    USE_VOLUME,
    USE_VOL_10D,
    USE_VOL_30D,
    USE_VOL_90D
)


def run_sweep(resume_training: bool = False, redo_training: bool = False):
    grid = itertools.product(
        HIST_WINDOWS,
        PV_HIST_WINDOWS,
        ACTION_HIST_WINDOWS,
        EMBEDDING_DIMS,
        NUM_EPISODES,
        VALIDATION_EVERY,
        RESUME_EVERY,
        ALPHA,
        BETA,
        GAMMA,
        TAU,
        NOISE_SCALE,
        WARM_UP,
        SHARPE_SCALING,
        SHARPE_SCALING_MONTHLY,
        SHARPE_SCALING_YEARLY,
        THETA_MAX,
        INITIAL_PV,
        LABEL_LATEST,
        LABEL_BEST_VAL,
        LABEL_FINAL,
        LAYER1_SIZE,
        LAYER2_SIZE,
        ASSET_LIST,
        W_VOL_10D,
        W_VOL_30D,
        W_VOL_90D,
        USE_VOLUME,
        USE_VOL_10D,
        USE_VOL_30D,
        USE_VOL_90D
    )

    for (
        hw, pw, aw, emb,
        num_eps, val_every, resume_every,
        alpha, beta, gamma, tau, noise, warm_up,
        sharpe_scaling, sharpe_scaling_monthly, sharpe_scaling_yearly, theta_max, initial_pv,
        lbl_latest, lbl_best, lbl_final,
        layer1_size, layer2_size, asset_list,
        w_vol_10d, w_vol_30d, w_vol_90d,
        use_volume, use_vol_10d, use_vol_30d, use_vol_90d
    ) in grid:
        run_id = (
            f"hw{hw}_pw{pw}_aw{aw}_emb{emb}"
            f"_l1{layer1_size}_l2{layer2_size}"
            f"_ss{sharpe_scaling}_tm{theta_max}"
        )
        print(f"\n=== SWEEP RUN: {run_id} ===")

        # initialize W&B
        config = {
            "hist_window": hw, "pv_hist_window": pw,
            "action_hist_window": aw, "embedding_dim": emb,
            "num_episodes": num_eps, "validation_every": val_every,
            "resume_every": resume_every, "alpha": alpha,
            "beta": beta, "gamma": gamma, "tau": tau,
            "noise_scale": noise, "warm_up": warm_up,
            "layer1_size": layer1_size, "layer2_size": layer2_size,
            "sharpe_scaling": sharpe_scaling, "sharpe_scaling_monthly": sharpe_scaling_monthly,
            "sharpe_scaling_yearly": sharpe_scaling_yearly,"theta_max": theta_max,
            "initial_pv": initial_pv, "asset_list": asset_list,
            "w_vol_10d": w_vol_10d, "w_vol_30d": w_vol_30d,
            "w_vol_90d": w_vol_90d, "use_volume": use_volume,
            "use_vol_10d": use_vol_10d, "use_vol_30d": use_vol_30d,
            "use_vol_90d": use_vol_90d
        }
        wandb.init(project="robo-advisor", name=run_id,
                   config=config, save_code=True)
        run_start = time.perf_counter()

        asset_list = ASSET_LIST
        # per-run logger & data loaders
        log_dir = os.path.join("logs", run_id)
        os.makedirs(log_dir, exist_ok=True)
        logger = LogManager(log_dir=log_dir, flush_every=1000)
        train_loader = make_loader("train", hw, emb, asset_list)
        val_loader = make_loader("validation", hw, emb, asset_list)
        test_loader = make_loader("test", hw, emb, asset_list)

        # environment and agent
        env = PortfolioEnv(
            hist_window=hw, pv_hist_window=pw,
            action_hist_window=aw,
            sharpe_scaling=sharpe_scaling,
            sharpe_scaling_monthly=sharpe_scaling_monthly,
            sharpe_scaling_yearly=sharpe_scaling_yearly,
            theta_max=theta_max,
            initial_pv=initial_pv,
            asset_list=asset_list,
            use_volume=use_volume,use_vol_10d=use_vol_10d,use_vol_30d=use_vol_30d,use_vol_90d=use_vol_90d
        )
        env.set_data_loader(train_loader)

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        n_assets = env.prices.shape[-1]
        agent = Agent(
            alpha=alpha, beta=beta, input_dims=obs_dim,
            tau=tau, env=env, gamma=gamma,
            update_actor_interval=val_every,
            warm_up=warm_up, n_actions=act_dim,
            n_assets=n_assets, max_size=200_000,
            layer1_size=layer1_size, layer2_size=layer2_size,
            batch_size=40, noise=noise
        )
        wandb.watch(agent.actor, log="all", log_freq=100)
        wandb.watch(agent.critic_1, log="all", log_freq=100)
        wandb.watch(agent.critic_2, log="all", log_freq=100)

        best_val_sharpe = -float("inf")
        if resume_training or redo_training:
            agent.load_models(label=LABEL_LATEST)
            print(f"[{run_id}] Loaded latest checkpoint")
            start_ep = (agent.current_episode + 1 if resume_training else 1)
        else:
            start_ep = 1

        # training + validation loop
        train_phase_start = time.perf_counter()
        for ep in range(start_ep, num_eps + 1):
            ep_start = time.perf_counter()
            tr_reward, tr_sharpe, tr_c_loss, tr_a_loss = train_one_episode(
                agent, env, train_loader, logger, ep)
            ep_time = time.perf_counter() - ep_start
            print(f"[Train][{run_id}] Ep {ep} — Reward {tr_reward:.4f}, "
                  f"Sharpe {tr_sharpe:.4f} (took {ep_time:.2f}s)")

            wandb.log({
                "train/episode": ep,
                "train/reward": tr_reward,
                "train/sharpe": tr_sharpe,
                "train/critic_loss":  tr_c_loss,
                "train/actor_loss":   tr_a_loss,
                "train/episode_time_s": ep_time,
            }, step=ep)

            # validation every `val_every` episodes
            if ep % val_every == 0:
                val_start = time.perf_counter()
                agent.save_models(label=lbl_latest)
                val_sh, val_r = evaluate(
                    agent, env, val_loader, logger,
                    phase="validation", episode=ep)
                val_time = time.perf_counter() - val_start
                print(f"[Val][{run_id}] Ep {ep} — Reward {val_r:.4f}, "
                      f"Sharpe {val_sh:.4f} (took {val_time:.2f}s)")

                wandb.log({
                    "validation/episode": ep,
                    "validation/reward": val_r,
                    "validation/sharpe": val_sh,
                    "validation/time_s": val_time,
                }, step=ep)

                if val_sh > best_val_sharpe:
                    best_val_sharpe = val_sh
                    agent.save_models(label=lbl_best)
                    print(f"  ▶ New best_val at Ep {ep}: {val_sh:.4f}")

        # after loop: log phase time
        train_phase_time = time.perf_counter() - train_phase_start
        print(f"[Train][{run_id}] finished {num_eps} eps in "
              f"{train_phase_time:.2f}s")
        wandb.log({"train/phase_time_s": train_phase_time})

        # final test & wrap-up...
        agent.load_models(label=lbl_best)
        test_start = time.perf_counter()
        test_sh, test_reward = evaluate(
            agent, env, test_loader, logger, phase="test")
        test_time = time.perf_counter() - test_start
        print(f"[Test][{run_id}] Reward {test_reward:.4f}, "
              f"Sharpe {test_sh:.4f} (took {test_time:.2f}s)")
        wandb.log({
            "test/reward": test_reward,
            "test/sharpe": test_sh,
            "test/time_s": test_time
        })

        wandb.finish()

        # 1) Ensure your reports/ folder exists
        reports_dir = "reports"
        os.makedirs(reports_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_fname = os.path.join(reports_dir, f"all_logs_{run_id}_{timestamp}.xlsx")

        # 2) Combine CSVs into that XLSX
        logger.finalize(out_fname)

        # 3) Now append your hyperparams sheet (transposed)
        hyperparams = {
            "hist_window":        hw,
            "pv_hist_window":     pw,
            "action_hist_window": aw,
            "embedding_dim":      emb,
            "num_episodes":       num_eps,
            "validation_every":   val_every,
            "resume_every":       resume_every,
            "alpha":              alpha,
            "beta":               beta,
            "gamma":              gamma,
            "tau":                tau,
            "noise_scale":        noise,
            "warm_up":            warm_up,
            "layer1_size":        layer1_size,
            "layer2_size":        layer2_size,
            "sharpe_scaling":     sharpe_scaling,
            "sharpe_scaling_monthly": sharpe_scaling_monthly,
            "sharpe_scaling_yearly":  sharpe_scaling_yearly,
            "theta_max":          theta_max,
            "initial_pv":         initial_pv,
            "label_latest":       lbl_latest,
            "label_best_val":     lbl_best,
            "label_final":        lbl_final,
            "data_path":          DATA_PATH,
        }
        # build a two-column DataFrame: parameter | value
        df_params = (pd.DataFrame.from_dict(hyperparams, orient='index', columns=['value'])
                        .reset_index()
                        .rename(columns={'index': 'parameter'}))

        # 4) Append sheet
        with pd.ExcelWriter(out_fname, engine="openpyxl", mode="a") as writer:
            df_params.to_excel(writer, sheet_name="hyperparams", index=False)

        # 5) (Optional) clean up the CSVs
        for phase in ("train","validation","test"):
            try:
                os.remove(os.path.join(log_dir, f"{phase}_logs.csv"))
            except OSError:
                pass

        print(f"[Sweep] Finished {run_id} → {out_fname}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Grid‑search runner with optional resume/redo."
    )
    parser.add_argument(
        "--resume_training",
        action="store_true",
        help="Load the ‘latest’ checkpoint at the start of each run"
    )
    parser.add_argument(
        "--redo_training",
        action="store_true",
        help="Wipe training history and start from that same checkpoint"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    run_sweep(resume_training=args.resume_training,
              redo_training=args.redo_training)
    print("Sweep completed.")