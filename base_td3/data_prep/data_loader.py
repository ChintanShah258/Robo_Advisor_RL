import numpy as np
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, df, hist_window, n_embeddings, asset_list, initial_pv=1000):
        """
        df            : DataFrame containing exactly the columns:
                        ['Dates']
                        + price_cols
                        + volume_cols
                        + vol_10d_cols
                        + vol_30d_cols
                        + vol_90d_cols
                        + embed_cols (last n_embeddings columns)
                        + reward/meta columns (mask_meta, year, month, etc.)
        hist_window   : number of past days to include in each sample
        n_embeddings  : number of embedding dimensions at end of df
        asset_list    : list of asset symbols, e.g. ['ECL','NEM','APD']
        initial_pv    : starting portfolio‐value (unused)
        """
        self.df = df.reset_index(drop=True)
        self.hist_window = hist_window
        self.n_embed = n_embeddings
        self.init_pv = initial_pv
        self.asset_list = asset_list
        self.n_assets = len(asset_list)

        # build column groups. Currently we are taking log_sd standardized (log(returns/prices)) values
        # but can change this to raw prices if needed. Jusr remove log_sd fro the suffix here.
        price_cols_raw   = [f"{sym}"          for sym in asset_list]
        price_cols     = [f"{sym}_log_sd"    for sym in asset_list]
        volume_cols  = [f"{sym}_Volume_log_sd"   for sym in asset_list]
        vol_10d_cols  = [f"{sym}_Vol_10D_log_sd"  for sym in asset_list]
        vol_30d_cols  = [f"{sym}_Vol_30D_log_sd"  for sym in asset_list]
        vol_90d_cols  = [f"{sym}_Vol_90D_log_sd"  for sym in asset_list]
        embed_cols   = list(self.df.columns[-n_embeddings:])

        # extract arrays
        self.prices_raw              = self.df[price_cols_raw].values
        self.prices                  = self.df[price_cols].values
        self.volume                  = self.df[volume_cols].values
        self.vol_10d                 = self.df[vol_10d_cols].values
        self.vol_30d                 = self.df[vol_30d_cols].values
        self.vol_90d                 = self.df[vol_90d_cols].values
        self.embeds                  = self.df[embed_cols].values
        self.arr_mask_meta           = self.df['mask_meta'].values

        # full‐length meta/target arrays
        self.arr_date                    = self.df['Dates'].values
        self.arr_year                    = self.df['year'].values
        self.arr_month                   = self.df['month'].values
        self.arr_months_in_current_year  = self.df['months_in_current_year'].values
        self.arr_week_of_month           = self.df['week_of_month'].values
        self.arr_weeks_in_current_month  = self.df['weeks_in_current_month'].values
        self.arr_week_of_year            = self.df['week_of_year'].values
        self.arr_weeks_in_current_year   = self.df['weeks_in_current_year'].values
        self.arr_max_weeks_in_year       = self.df['max_weeks_in_year'].values
        self.arr_days_in_week            = self.df['days_in_week'].values
        self.arr_days_in_month           = self.df['days_in_month'].values
        self.arr_days_in_year            = self.df['days_in_year'].values
        self.arr_calendar_days_in_year   = self.df['calendar_days_in_year'].values
        self.arr_yr_rank                 = self.df['year_rank'].values
        self.arr_et_w                    = self.df['weekly_target_(daily)_cumulative'].values
        self.arr_et_m                    = self.df['monthly_target_(daily)_cumulative'].values
        self.arr_et_y                    = self.df['annual_target_(daily)_cumulative'].values

    def __len__(self):
        # number of samples = total rows minus warm‐up history
        return len(self.df) - self.hist_window

    def __getitem__(self, idx):
        i = idx + self.hist_window

        if self.hist_window > 0:
            start = i - self.hist_window
            end   = i

            prices_raw_window = self.prices_raw[start:end, :]  # (hist_window, n_assets)
            prices_window  = self.prices[start:end, :]        # (hist_window, n_assets)
            volume_window  = self.volume[start:end, :]
            vol_10d_window  = self.vol_10d[start:end, :]
            vol_30d_window  = self.vol_30d[start:end, :]
            vol_90d_window  = self.vol_90d[start:end, :]
            embeds_window  = self.embeds[start:end, :]        # (hist_window, n_embeddings)

            # mask the *first* hist_window-1 rows, so the last row is always live
            if self.hist_window > 1:
                mask_meta_window = self.arr_mask_meta[start:end]
            else:
                mask_meta_window = np.empty((0,), dtype=self.arr_mask_meta.dtype)

        else:
            # hist_window == 0 → all windows should be empty
            prices_raw_window = np.empty((0, self.n_assets), dtype=self.prices_raw.dtype)
            prices_window  = np.empty((0, self.n_assets),   dtype=self.prices.dtype)
            volume_window  = np.empty((0, self.n_assets),   dtype=self.volume.dtype)
            vol_10d_window  = np.empty((0, self.n_assets),   dtype=self.vol_10d.dtype)
            vol_30d_window  = np.empty((0, self.n_assets),   dtype=self.vol_30d.dtype)
            vol_90d_window  = np.empty((0, self.n_assets),   dtype=self.vol_90d.dtype)
            embeds_window  = np.empty((0, self.n_embed),    dtype=self.embeds.dtype)
            mask_meta_window = np.empty((0,), dtype=self.arr_mask_meta.dtype)

        return {
            'i': i,
            'prices_raw': prices_raw_window,
            'prices': prices_window,
            'volume': volume_window,
            'vol_10d': vol_10d_window,
            'vol_30d': vol_30d_window,
            'vol_90d': vol_90d_window,
            'embeds': embeds_window,
            'mask_meta': mask_meta_window,

            'arr_date': self.arr_date[i],
            'arr_year': self.arr_year[i],
            'arr_month': self.arr_month[i],
            'arr_months_in_current_year': self.arr_months_in_current_year[i],
            'arr_week_of_month': self.arr_week_of_month[i],
            'arr_weeks_in_current_month': self.arr_weeks_in_current_month[i],
            'arr_week_of_year': self.arr_week_of_year[i],
            'arr_weeks_in_current_year': self.arr_weeks_in_current_year[i],
            'arr_max_weeks_in_year': self.arr_max_weeks_in_year[i],
            'arr_days_in_week': self.arr_days_in_week[i],
            'arr_days_in_month': self.arr_days_in_month[i],
            'arr_days_in_year': self.arr_days_in_year[i],
            'arr_calendar_days_in_year': self.arr_calendar_days_in_year[i],
            'arr_yr_rank': self.arr_yr_rank[i],

            'arr_et_w': self.arr_et_w[i],
            'arr_et_m': self.arr_et_m[i],
            'arr_et_y': self.arr_et_y[i],
        }
