import gym, numpy as np
from gym import spaces
from collections import deque
import torch

class PortfolioEnv(gym.Env):
    def __init__(self, hist_window=21,pv_hist_window=5, action_hist_window=1, theta_max=3.0, 
                sharpe_scaling=0.01, sharpe_scaling_monthly=0.01,sharpe_scaling_yearly=0.01, 
                variance_scaling = 0.1,monthly_variance_scaling = 0.2, yearly_variance_scaling = 0.1,
                initial_pv=1000, da_ss = 0.2, use_volume=True, use_vol_10d=True, use_vol_30d=True,
                use_vol_90d=True, use_embeds=True):
        super().__init__()
        # hyperparams
        self.hist_window = hist_window
        self.pv_hist_window=pv_hist_window         
        self.action_hist_window=action_hist_window
        self.sharpe_scaling = sharpe_scaling
        self.sharpe_scaling_monthly = sharpe_scaling_monthly
        self.sharpe_scaling_yearly = sharpe_scaling_yearly
        self.variance_scaling = variance_scaling
        self.monthly_variance_scaling = monthly_variance_scaling
        self.yearly_variance_scaling = yearly_variance_scaling
        self.theta_max = theta_max
        self.initial_pv = initial_pv
        
        # which extra blocks to pack into obs:
        self.use_volume  = use_volume
        self.use_embeds  = use_embeds
        # volume & volatility
        self.use_vol_10d  = use_vol_10d
        self.use_vol_30d  = use_vol_30d
        self.use_vol_90d  = use_vol_90d
        
        # initial pacing baselines
        self.week_start_pv = self.month_start_pv = self.year_start_pv = self.initial_pv
        
        # make sure we always keep at least one step of PV/action history
        self.price_hist = hist_window
        self.embed_hist = hist_window
        self.pv_hist_len = pv_hist_window
        self.action_hist_len = action_hist_window
        
        # dual‐ascent step size
        self.da_ss = da_ss

        # Rolling‐Sharpe buffer
        self.recent_returns = deque(maxlen=self.hist_window)
        
        # placeholders for data; will be set by set_data_loader()
        self.original_loader = None
        self._make_spaces()

    def _make_spaces(self):
        self.action_space = None
        self.observation_space = None

    def _unpack_current(self):
        single = {}
        for k, v in self.current_batch.items():
            single[k] = v        # keep the full batch
        self._unpack_batch(single)

    def set_data_loader(self, loader):
        self.original_loader = loader
        self.loader = iter(loader)
        
        # fetch the first batch of B windows
        self.current_batch = next(self.loader)
        self.B = self.current_batch['prices'].shape[0]
        
        # store total length of your df
        self.df_length = len(loader.dataset.df)

        # unpack the very first window (pointer=0)
        self._unpack_current()
        
        # make per‑env pacing baselines (arrays of shape (B,))
        self.week_start_pv = np.full((self.B,), self.initial_pv, dtype=np.float32)
        self.month_start_pv = np.full((self.B,), self.initial_pv, dtype=np.float32)
        self.year_start_pv = np.full((self.B,), self.initial_pv, dtype=np.float32)

        # After fetching the first batch:
        n_assets = self.current_batch['prices'].shape[-1]
        n_embeds = self.current_batch['embeds'].shape[-1]

        # Action: 2 * n_assets weights + lambda + theta
        action_dim = 2 * n_assets + 2
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(action_dim,), dtype=np.float32
        )

        # now compute obs_dim
        obs_dim = 0
        # 1) prices
        obs_dim += n_assets * self.hist_window
        # 2) optional volume & vols
        if self.use_volume:  obs_dim += n_assets * self.hist_window
        if self.use_vol_10d:  obs_dim += n_assets * self.hist_window
        if self.use_vol_30d:  obs_dim += n_assets * self.hist_window
        if self.use_vol_90d:  obs_dim += n_assets * self.hist_window
        # 3) embeddings
        if self.use_embeds: obs_dim += n_embeds * self.hist_window
        # 4) pv + action history
        obs_dim += self.pv_hist_len
        obs_dim += self.action_hist_len * action_dim
        # 5) the 6 extra dual/ascent & target features
        obs_dim += 6

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        print(f"[Env] B={self.B}, n_assets={n_assets}, n_embeds={n_embeds}")
        print(f"[Env] action_dim={action_dim}, obs_dim={obs_dim}")

    def _unpack_batch(self, batch):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            setattr(self, k, v)
        # Now you have self.prices, self.embeds, and:
        # self.mask_meta of shape (B, hist_window)
        self.mask_meta_window = self.mask_meta
        self.i = batch['i']  # shape (B,)

    def reset(self):
        B = self.B
        self._first_step = True

        # 1) PV & action history: use the user‐specified windows (can be zero!)
        self.pv_history     = np.full((B, self.pv_hist_len), self.initial_pv, dtype=np.float32)
        action_dim = self.action_space.shape[0]
        self.action_history = np.zeros((B, self.action_hist_len, action_dim), dtype=np.float32)

        # 2) Other per‐step state
        self.current_step = np.full((B,), self.hist_window, dtype=int)
        self.prev_pv      = np.full((B,), self.initial_pv, dtype=np.float32)
        self.pv           = np.full((B,), self.initial_pv, dtype=np.float32)
        self.new_pv       = np.full((B,), self.initial_pv, dtype=np.float32)

        # 3) Pacing baselines
        self.week_start_pv  = np.full((B,), self.initial_pv, dtype=np.float32)
        self.month_start_pv = np.full((B,), self.initial_pv, dtype=np.float32)
        self.year_start_pv  = np.full((B,), self.initial_pv, dtype=np.float32)

        # 4) Dual multipliers
        self.lgr_mult_weekly  = np.zeros((B,), dtype=np.float32)
        self.lgr_mult_monthly = np.zeros((B,), dtype=np.float32)
        self.lgr_mult_annual  = np.zeros((B,), dtype=np.float32)

        # 5) Rolling Sharpe buffer
        self.recent_returns = np.zeros((B, self.hist_window), dtype=np.float32)
        self.rr_index       = np.zeros((B,), dtype=int)
        # ── per-env monthly/year buffers ──
        self._current_month_returns = [deque() for _ in range(self.B)]
        self.monthly_sharpes       = [[]    for _ in range(self.B)]
        self._current_year_returns = [deque() for _ in range(self.B)]
        self.yearly_sharpes       = [[]    for _ in range(self.B)]
        self.monthly_vars         = [[]    for _ in range(self.B)]
        self.yearly_vars          = [[]    for _ in range(self.B)]

        # 6) Reload the data iterator
        self.loader = iter(self.original_loader)
        self.current_batch = next(self.loader)
        self._unpack_current()

        # 7) Store previous calendar markers
        self.prev_arr_week  = self.arr_week_of_year.copy()
        self.prev_arr_month = self.arr_month.copy()
        self.prev_arr_yr_rank = self.arr_yr_rank.copy()

        print("unpacked prices.shape:", self.prices.shape)
        print("unpacked embeds.shape:", self.embeds.shape)

        # 8) Initialize all logging fields so step=0 logs are valid
        self.r_week       = np.zeros((B,), dtype=np.float32)
        self.r_month      = np.zeros((B,), dtype=np.float32)
        self.r_year       = np.zeros((B,), dtype=np.float32)
        self.pen_w        = np.zeros((B,), dtype=np.float32)
        self.pen_m        = np.zeros((B,), dtype=np.float32)
        self.pen_y        = np.zeros((B,), dtype=np.float32)
        self.sharpe       = np.zeros((B,), dtype=np.float32)
        self.r_env        = np.zeros((B,), dtype=np.float32)
        self.daily_return = np.zeros((B,), dtype=np.float32)
        self.mean_r      = np.zeros((B,), dtype=np.float32)
        self.std_r       = np.zeros((B,), dtype=np.float32)
        self.variance_r  = np.zeros((B,), dtype=np.float32)
        self.episode_rewards = np.zeros((B,), dtype=float)
        # at start of reset(), after B = self.B
        self.weights = np.zeros((B, self.prices.shape[-1]), dtype=np.float32)

        # Setting this to raw prices because we use raw prices to calcualte asset daily returns
        # as part of Portfolio Value Calcualtions. We feed in Standardized (log(retruns)) to the
        # Observation Space (prices). Can change back to raw values in DataProcessor.py
        # file. If you do that then it shuold be set to self.prices here instead of self.prices_raw.
        self.prev_prices = self.prices_raw[:, -1, :].copy()

        # 9) Return first observation
        return self._get_obs()

    def _get_obs(self):
        B = self.B
        # 1) price history
        parts = [ self.prices.reshape(B, -1) ]

        # 2) optional volume & vols
        if self.use_volume:
            parts.append(self.volume.reshape(B, -1))
        if self.use_vol_10d:
            parts.append(self.vol_10d.reshape(B, -1))
        if self.use_vol_30d:
            parts.append(self.vol_30d.reshape(B, -1))
        if self.use_vol_90d:
            parts.append(self.vol_90d.reshape(B, -1))

        # 3) embeddings
        if self.use_embeds:
            parts.append(self.embeds.reshape(B, -1))
        

        # 4) pv & action history
        parts.append(self.pv_history)                              # (B, pv_hist_len)
        parts.append(self.action_history.reshape(B, -1))

        # 5) extras: et_w, et_m, et_y, lgr_mult_weekly, month, annual
        et_w, et_m, et_y = self.arr_et_w, self.arr_et_m, self.arr_et_y
        lw, lm, la        = self.lgr_mult_weekly, self.lgr_mult_monthly, self.lgr_mult_annual
        extras = np.stack([et_w, et_m, et_y, lw, lm, la], axis=1)
        parts.append(extras)

        return np.concatenate(parts, axis=1).astype(np.float32)
   
    def _compute_step(self, action):
        B        = self.B
        H        = self.hist_window
        n_assets = self.prices.shape[-1]
        eps      = 1e-8

        # 1) Decode action → portfolio weights (sum to 1)
        w_base       = action[:, :n_assets]
        w_risk       = action[:, n_assets:2*n_assets]
        amount_risky = action[:, 2*n_assets]
        theta_risky  = action[:, 2*n_assets + 1]

        alloc_base   = (1 - amount_risky)[:, None] * w_base
        alloc_risky  = (amount_risky)[:, None] * w_risk
        weights      = alloc_base + alloc_risky
        sum_w        = weights.sum(axis=1, keepdims=True)
        sum_w[sum_w < eps] = 1.0
        weights     /= sum_w

        # 2) Compute daily asset returns, guarding div‑by‑zero
        yesterday = self.prev_prices.copy()        # (B, n_assets)
        today     = self.prices_raw[:, -1, :].copy()   # (B, n_assets)
        rets      = np.zeros_like(today, dtype=np.float32)
        mask      = yesterday > eps
        rets[mask] = today[mask] / yesterday[mask] - 1.0
        self.prev_prices = today.copy()

        # 3) Portfolio return & new PV
        portfolio_ret = (weights * rets).sum(axis=1)
        portfolio_ret = np.nan_to_num(portfolio_ret, nan=0.0, posinf=0.0, neginf=0.0)
        new_pv        = self.pv * (1 + portfolio_ret)
        new_pv        = np.clip(new_pv, 1e-3, 1e6)
        self.new_pv   = new_pv.copy()
        self.pv       = new_pv

        daily_return = portfolio_ret.copy()

        # 4) Pacing returns with np.divide(out, where)
        r_week  = np.zeros_like(new_pv, dtype=np.float32)
        cond_wk = self.week_start_pv > eps
        np.divide(new_pv, self.week_start_pv, out=r_week, where=cond_wk)
        r_week[cond_wk] -= 1.0

        r_month  = np.zeros_like(new_pv, dtype=np.float32)
        cond_mo  = self.month_start_pv > eps
        np.divide(new_pv, self.month_start_pv, out=r_month, where=cond_mo)
        r_month[cond_mo] -= 1.0

        r_year  = np.zeros_like(new_pv, dtype=np.float32)
        cond_yr = self.year_start_pv > eps
        np.divide(new_pv, self.year_start_pv, out=r_year, where=cond_yr)
        r_year[cond_yr] -= 1.0
        
        # 5) Penalties & net env reward
        pen_w = self.lgr_mult_weekly  * np.maximum(0.0, self.arr_et_w - r_week)
        pen_m = self.lgr_mult_monthly * np.maximum(0.0, self.arr_et_m - r_month)
        pen_y = self.lgr_mult_annual  * np.maximum(0.0, self.arr_et_y - r_year)
        pen_w, pen_m, pen_y = (
            np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            for x in (pen_w, pen_m, pen_y)
        )
        r_env = daily_return - (pen_w + pen_m + pen_y)

        # 6) Rolling Sharpe bonus
        idx = self.rr_index
        self.recent_returns[:, idx] = daily_return
        self.rr_index = (idx + 1) % max(1, H)
        # also accumulate for monthly Sharpe
        self._current_month_returns.append(daily_return)

        if H > 1:
            mean_r = self.recent_returns.mean(axis=1)
            std_r  = self.recent_returns.std(axis=1, ddof=1)
            variance_r = self.recent_returns.var(axis=1, ddof=1)
            variance_r = np.nan_to_num(variance_r, nan=0.0, posinf=0.0, neginf=0.0)
            
            std_r = np.nan_to_num(std_r, nan=0.0, posinf=0.0, neginf=0.0)
            sharpe = np.zeros_like(mean_r, dtype=np.float32)
            valid  = std_r > eps
            sharpe[valid] = mean_r[valid] / std_r[valid]
        else:
            sharpe = np.zeros_like(daily_return, dtype=np.float32)

        # store them on self for logging later
        self.mean_r = mean_r
        self.std_r  = std_r
        self.variance_r = variance_r
        
        sharpe = np.nan_to_num(sharpe, nan=0.0, posinf=0.0, neginf=0.0)

        # 8) Dual‐ascent updates on calendar boundaries
        if self._first_step:
            self._first_step = False
        else:
            # weekly
            mask_w = (self.arr_week_of_year != self.prev_arr_week)
            r_end_w = np.zeros_like(new_pv, dtype=np.float32)
            np.divide(self.pv, self.week_start_pv, out=r_end_w, where=cond_wk)
            r_end_w[cond_wk] -= 1.0
            self.lgr_mult_weekly[mask_w] += self.da_ss * (self.arr_et_w[mask_w] - r_end_w[mask_w])
            self.week_start_pv[mask_w]   = new_pv[mask_w]

            # 1) figure out which copies have rolled over into a new month
            mask_m = (self.arr_month != self.prev_arr_month)

            # 2) first, append today’s daily_return to each env’s month-buffer
            for i in range(self.B):
                self._current_month_returns[i].append(daily_return[i])

            # 3) for each env that hit month-boundary, compute sharpe and clear its buffer
            for i in np.where(mask_m)[0]:
                rets = np.array(self._current_month_returns[i], dtype=np.float32)
                if rets.size > 1:
                    sharpe_m = rets.mean() / (rets.std() + 1e-8)
                    var_m = rets.var()           # monthly variance
                else:
                    sharpe_m = 0.0
                    var_m = 0.0
                self.monthly_vars[i].append(var_m)  # store in a parallel buffer list
                self.monthly_sharpes[i].append(sharpe_m)
                self._current_month_returns[i].clear()

            # 4) now existing pacing penalty for month
            r_end_m = np.zeros_like(new_pv, dtype=np.float32)
            np.divide(self.pv, self.month_start_pv, out=r_end_m, where=cond_mo)
            r_end_m[cond_mo] -= 1.0
            self.lgr_mult_monthly[mask_m] += self.da_ss * (self.arr_et_m[mask_m] - r_end_m[mask_m])
            self.month_start_pv[mask_m]   = new_pv[mask_m]

            # yearly
            mask_y = (self.arr_yr_rank != self.prev_arr_yr_rank)
            # append daily returns
            for i in range(self.B):
                self._current_year_returns[i].append(daily_return[i])
            for i in np.where(mask_y)[0]:
                rets = np.array(self._current_year_returns[i], dtype=np.float32)
                if rets.size > 1:
                    sharpe_y = rets.mean() / (rets.std() + 1e-8)
                    var_y = rets.var()
                else:
                    sharpe_y = 0.0
                    var_y = 0.0                
                self.yearly_vars[i].append(var_y)
                self.yearly_sharpes[i].append(sharpe_y)
                self._current_year_returns[i].clear()

            # now accumulate this day’s return into the next-year buffer
            self._current_year_returns.append(daily_return)
            # ── end “yearly Sharpe” insert ──
        
            r_end_y = np.zeros_like(new_pv, dtype=np.float32)
            np.divide(self.pv, self.year_start_pv, out=r_end_y, where=cond_yr)
            r_end_y[cond_yr] -= 1.0
            self.lgr_mult_annual[mask_y] += self.da_ss * (self.arr_et_y[mask_y] - r_end_y[mask_y])
            self.year_start_pv[mask_y]   = new_pv[mask_y]

        # 7) Total reward
        sharpe_monthly = np.array([buffs[-1] if len(buffs)>0 else 0.0 
                                 for buffs in self.monthly_sharpes], dtype=np.float32) # shape (B,)
        sharpe_yearly  = np.array([buffs[-1] if len(buffs)>0 else 0.0 
                                 for buffs in self.yearly_sharpes], dtype=np.float32) # shape (B,)
        
        variance_monthly = np.array([buffs[-1] if len(buffs)>0 else 0.0 
                                 for buffs in self.monthly_vars],dtype=np.float32)
        variance_yearly = np.array([buffs[-1] if len(buffs)>0 else 0.0
                                 for buffs in self.yearly_vars],dtype=np.float32)

        reward = (
            r_env
            + self.sharpe_scaling         * sharpe
            + self.sharpe_scaling_monthly * sharpe_monthly
            + self.sharpe_scaling_yearly  * sharpe_yearly
            - self.variance_scaling       * variance_r
            - self.monthly_variance_scaling * variance_monthly
            - self.yearly_variance_scaling  * variance_yearly
        )
        reward = np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.episode_rewards += reward
        
        # 9) Update previous markers & histories...
        self.prev_arr_week    = self.arr_week_of_year.copy()
        self.prev_arr_month   = self.arr_month.copy()
        self.prev_arr_yr_rank = self.arr_yr_rank.copy()

        self.pv_history     = np.roll(self.pv_history, -1, axis=1)
        self.pv_history[:, -1]     = new_pv
        self.action_history = np.roll(self.action_history, -1, axis=1)
        self.action_history[:, -1, :] = action

        # 10) Store for logging & step counter
        self.daily_return = daily_return
        self.r_week       = r_week
        self.r_month      = r_month
        self.r_year       = r_year
        self.pen_w        = pen_w
        self.pen_m        = pen_m
        self.pen_y        = pen_y
        self.sharpe       = sharpe
        self.r_env        = r_env
        self.new_pv       = new_pv
        self.current_step += 1

        # 11) Build observation & return
        obs = self._get_obs()
        return obs, reward

    # def step(self, action):
    #     # 1) convert action to numpy
    #     if hasattr(action, "detach"):
    #         action = action.detach().cpu().numpy()

    #     # 2) compute obs & reward
    #     obs, reward = self._compute_step(action)

    #     # 3) done when i == len(df)
    #     done = (self.i >= self.df_length)

    #     # 4) advance to the next rolling batch of windows
    #     try:
    #         print("Advancing loader: i =", self.i, "df_length-1 =", self.df_length-1)
    #         self.current_batch = next(self.loader)
    #         self._unpack_batch(self.current_batch)
    #     except StopIteration:
    #         # exhausted
    #         # no more data → mark done for all env copies
    #         done = np.ones_like(done, dtype=bool)
    #         # optionally clear current_batch to avoid re‑use
    #         self.current_batch = None

    #     return obs, reward, done, {}
    
    def step(self, action):
        # 1) ensure numpy
        if hasattr(action, "detach"):
            action = action.detach().cpu().numpy()

        # 2) try to advance loader; if exhausted, we still compute the last step, then finish
        try:
            self.current_batch = next(self.loader)
            self._unpack_batch(self.current_batch)
            data_exhausted = False
        except StopIteration:
            data_exhausted = True

        # 3) compute obs & reward
        obs, reward = self._compute_step(action)

        # 4) determine done: either we ran out of data *or* some other condition
        done = np.ones_like(self.pv, dtype=bool) if data_exhausted else np.zeros_like(self.pv, dtype=bool)

        return obs, reward, done, {}

