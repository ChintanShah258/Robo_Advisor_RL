#!/usr/bin/env python3
"""
This script runs a hyperparameter sweep using Weights & Biases Sweeps.
Usage:
  # 1) register the sweep (once):
  wandb sweep sweep_final.yml

  # 2) launch agents (each will call train()):
  wandb agent chintanshah4-carleton-university/robo-advisor/SWEEP_ID
"""

import os
import argparse
import time
from datetime import datetime
import numpy as np

import wandb
from main import (
    train_one_episode,
    evaluate,
    make_loader,
    PortfolioEnv,
    Agent,
    LogManager,
)

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def train(resume_training: bool, redo_training: bool, args):
    # --- initialize W&B run with explicit style ---
    run = wandb.init(
        project="robo-advisor",              # your project name
        name=f"agent-{int(time.time())}",    # or any naming scheme
        config=args,                         # all of our hyperparameters
        save_code=True,
        reinit=True
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
    wandb.define_metric("critic/step")
    wandb.define_metric("critic/loss",   step_metric="critic/step")
    wandb.define_metric("actor/step")
    wandb.define_metric("actor/loss",   step_metric="actor/step")

    
    print(f"Started W&B run name={run.name}, id={run.id}")
    config = wandb.config

    # unpack from config
    hw    = config.hist_window
    pw    = config.pv_hist_window
    aw    = config.action_hist_window
    emb   = config.embedding_dim
    num_eps     = config.num_episodes
    validation_every   = config.validation_every
    alpha       = config.alpha
    beta        = config.beta
    gamma       = config.gamma
    tau         = config.tau
    noise       = config.noise_scale
    warm_up     = config.warm_up
    sharpe_scaling         = config.sharpe_scaling
    sharpe_scaling_monthly = config.sharpe_scaling_monthly
    sharpe_scaling_yearly  = config.sharpe_scaling_yearly
    variance_scaling = config.variance_scaling
    monthly_variance_scaling = config.monthly_variance_scaling
    yearly_variance_scaling  = config.yearly_variance_scaling
    theta_max   = config.theta_max
    initial_pv  = config.initial_pv
    layer1_size = config.layer1_size
    layer2_size = config.layer2_size
    asset_list  = config.asset_list
    if isinstance(asset_list, str):
        asset_list = asset_list.split(",")
    use_volume  = config.use_volume
    use_embeds  = config.use_embeds
    use_vol_10d = config.use_vol_10d
    use_vol_30d = config.use_vol_30d
    use_vol_90d = config.use_vol_90d
    update_every = config.update_every
    gradient_steps = config.gradient_steps
    da_ss = config.da_ss
    update_actor_interval = config.update_actor_interval

    # set up logging directory
    run_id = run.id
    log_dir = os.path.join("logs", run_id)
    os.makedirs(log_dir, exist_ok=True)
    logger = LogManager(log_dir=log_dir, flush_every=1000)

    # data loaders
    train_loader = make_loader('train',      hw, emb, asset_list)
    val_loader   = make_loader('validation', hw, emb, asset_list)
    test_loader  = make_loader('test',       hw, emb, asset_list)

    # environment & agent
    env = PortfolioEnv(
        hist_window=hw,
        pv_hist_window=pw,
        action_hist_window=aw,
        sharpe_scaling=sharpe_scaling,
        sharpe_scaling_monthly=sharpe_scaling_monthly,
        sharpe_scaling_yearly=sharpe_scaling_yearly,
        variance_scaling = variance_scaling,
        monthly_variance_scaling = monthly_variance_scaling,
        yearly_variance_scaling  = yearly_variance_scaling,
        theta_max=theta_max,
        initial_pv=initial_pv,
        use_volume=use_volume,
        use_embeds=use_embeds,
        use_vol_10d=use_vol_10d,
        use_vol_30d=use_vol_30d,
        use_vol_90d=use_vol_90d,
        da_ss=da_ss,
    )
    env.set_data_loader(train_loader)

    obs_dim  = env.observation_space.shape[0]
    act_dim  = env.action_space.shape[0]
    n_assets = env.prices.shape[-1]

    agent = Agent(
        alpha=alpha, beta=beta,
        input_dims=obs_dim, tau=tau,
        env=env, gamma=gamma,
        update_actor_interval=update_actor_interval, warm_up=warm_up,
        n_actions=act_dim, n_assets=n_assets,
        max_size=250000, layer1_size=layer1_size,
        layer2_size=layer2_size, batch_size=40,
        noise=noise,
        T_max=config.T_max,        # from your sweep or config
        eta_min=config.eta_min     # from your sweep or config
    )

    wandb.watch(agent.actor,    log="all", log_freq=100)
    wandb.watch(agent.critic_1, log="all", log_freq=100)
    wandb.watch(agent.critic_2, log="all", log_freq=100)

    # optionally wipe or load checkpoints
    if redo_training:
        for label in ("latest","best_val","final"):
            path = os.path.join(agent.actor.checkpoint_dir, f"agent_{label}.pth")
            if os.path.isfile(path): os.remove(path)
    if resume_training:
        agent.load_models(label="latest")

    # training loop
    best_val_sharpe = -np.inf
    best_val_monthly = -np.inf
    
    for ep in range(1, num_eps+1):
        tr_reward, tr_sharpe, tr_monthly_sharpe, tr_yearly_sharpe, \
        tr_monthly_ret, tr_yearly_ret, tr_c_loss, tr_a_loss, tr_time = train_one_episode(
        agent, env, train_loader, logger, ep,update_every=update_every, gradient_steps=gradient_steps)
        # ── Log to W&B ──
        wandb.log({
            "train/episode":            ep,
            "train/reward":             tr_reward,
            "train/sharpe":             tr_sharpe,
            "train/avg_monthly_sharpe": tr_monthly_sharpe,
            "train/avg_yearly_sharpe":  tr_yearly_sharpe,
            "train/avg_monthly_return": tr_monthly_ret,
            "train/avg_yearly_return":  tr_yearly_ret,
            "train/critic_loss":        tr_c_loss,
            "train/actor_loss":         tr_a_loss,
            "train/episode_time_s":     tr_time,
        }, commit=False)

        if ep % validation_every == 0:
            val_start = time.perf_counter()
            agent.save_models(label="latest", current_episode=ep)
            print(f"[Main] Saved rolling ’latest’ at ep {ep}")
            vs, vr, vm, vy, vm_ret, vy_ret = evaluate(agent, env, val_loader, logger, phase="validation", episode=ep)
            val_time = time.perf_counter() - val_start
            print(f"[Val]   Ep {ep:>2} — Reward {vr:.4f}, Sharpe {vs:.4f},  (took {val_time:.2f}s)")
            
            # ── log to W&B here ──
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


    # ─── Final Test & Snapshot ────────────────────
    agent.load_models(label="best_val")
    test_start = time.perf_counter()
    ts, tr, tm, ty,tm_ret, ty_ret = evaluate(agent, env, test_loader, logger, 'test', None)
    test_time = time.perf_counter() - test_start
    print(f"[Test] — Reward {tr:.4f}, Sharpe {ts:.4f}  (took {test_time:.2f}s)")

    # ─── d) Log final test metrics ────────────────
    test_episode = 1 # Placeholder, as we only run one test episode outside if training loop
    wandb.log({
        "test/episode":           test_episode,
        "test/reward":            tr,
        "test/sharpe":            ts,
        "test/avg_monthly_sharpe": tm,
        "test/avg_yearly_sharpe":  ty,
        "test/avg_monthly_return": tm_ret,
        "test/avg_yearly_return":  ty_ret,
        "test/time_s":             test_time
    }, commit=False)

    wandb.log({}, step=ep, commit=True)
    
    agent.save_models(label="final")
    out_xlsx = os.path.join("reports", f"all_logs_{run_id}.xlsx")
    logger.finalize(out_xlsx)

    wandb.finish()

if __name__ == "__main__":
    p = argparse.ArgumentParser("W&B sweep runner")
    p.add_argument("--resume_training", type=str2bool, required=True)
    p.add_argument("--redo_training",   type=str2bool, required=True)
    p.add_argument("--hist_window",            type=int,   required=True)
    p.add_argument("--pv_hist_window",         type=int,   required=True)
    p.add_argument("--action_hist_window",     type=int,   required=True)
    p.add_argument("--embedding_dim",           type=int,   required=True)
    p.add_argument("--num_episodes",           type=int,   required=True)
    p.add_argument("--validation_every",       type=int,   required=True)
    p.add_argument("--alpha",                  type=float, required=True)
    p.add_argument("--beta",                   type=float, required=True)
    p.add_argument("--gamma",                  type=float, required=True)
    p.add_argument("--tau",                    type=float, required=True)
    p.add_argument("--noise_scale",            type=float, required=True)
    p.add_argument("--warm_up",                type=int,   required=True)
    p.add_argument("--sharpe_scaling",         type=float, required=True)
    p.add_argument("--sharpe_scaling_monthly", type=float, required=True)
    p.add_argument("--sharpe_scaling_yearly",  type=float, required=True)
    p.add_argument("--variance_scaling",       type=float, required=True)
    p.add_argument("--monthly_variance_scaling", type=float, required=True)
    p.add_argument("--yearly_variance_scaling",  type=float, required=True)
    p.add_argument("--theta_max",               type=float, required=True)
    p.add_argument("--initial_pv",              type=float, required=True)
    p.add_argument("--layer1_size",            type=int,   required=True)
    p.add_argument("--layer2_size",            type=int,   required=True)
    p.add_argument("--asset_list", type=lambda s: s.split(","),required=True,help="comma-separated list of assets")
    p.add_argument("--use_volume",    type=str2bool, required=True)
    p.add_argument("--use_embeds",    type=str2bool, required=True)
    p.add_argument("--use_vol_10d",   type=str2bool, required=True)
    p.add_argument("--use_vol_30d",   type=str2bool, required=True)
    p.add_argument("--use_vol_90d",   type=str2bool, required=True)
    p.add_argument("--update_every",    type=int, required=False, default=40)
    p.add_argument("--gradient_steps",  type=int, required=False, default=20)
    p.add_argument("--T_max",           type=int,   required=True, help="CosineAnnealing T_max")
    p.add_argument("--eta_min",         type=float, required=True, help="CosineAnnealing eta_min")
    p.add_argument("--da_ss",          type=float, required=True, help="Dual Ascent Lagrange Multiplier step size")
    p.add_argument('--update_actor_interval', type=int, required=True, help="Actor Update Frequency")
    
    args = p.parse_args()
    # convert Namespace to dict for config
    cfg = vars(args)

    # launch the W&B agent (assumes you've already run `wandb sweep sweep_final.yml`)
    # hand off to your train function
    train(
      resume_training=args.resume_training,
      redo_training=args.redo_training,
      args=vars(args)
    )
