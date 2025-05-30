# logging_utils.py
import openpyxl
import os
import csv
import torch
from typing import Optional

# All the fields we’ll ever log; order here determines CSV column order & Excel sheet order.
LOG_FIELDS = [
    'phase','episode','dataset_i','step','mask_meta','prices',
    'arr_date','arr_year','arr_month','arr_week_of_year','arr_weeks_in_current_year',
    'arr_week_of_month','arr_weeks_in_current_month','arr_months_in_current_year',
    'arr_days_in_week','arr_days_in_month','arr_days_in_year','arr_calendar_days_in_year',
    'arr_yr_rank','arr_et_w','arr_et_m','arr_et_y',
    'w_base','w_risky','amount_risky','theta',
    'pv','pv_weekly','pv_monthly','pv_yearly',
    'r_week','r_month','r_year','pen_w','pen_m','pen_y',
    'sharpe','sharpe_monthly','sharpe_yearly','r_env','daily_return','lagr_weekly','lagr_monthly','lagr_annual',
    'episode_rewards','new_pv','reward','mean_r','std_r',
    'variance_r','variance_monthly','variance_yearly','weights', 'new_weights',
]

class LogManager:
    def __init__(self,
                 log_dir: str = ".",
                 flush_every: int = 2000,
                 write_every: int =4000):
        """
        Streaming logger: writes one completed record at a time.
        - log_dir: where to write `{phase}_logs.csv`
        - flush_every: after this many rows, do a file.flush()
        """
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.flush_every = flush_every
        self.write_every = write_every
        self._write_buffer = { phase: [] for phase in ('train','validation','test') }

        self._files = {}
        self._writers = {}
        self._counters = {}       # how many rows written so far, per phase
        self._temp_buffer = {}    # holds the “before” rows until log_after is called

        for phase in ('train','validation','test'):
            path = os.path.join(log_dir, f"{phase}_logs.csv")
            f = open(path, "w", newline="")
            w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            w.writeheader()

            self._files[phase]   = f
            self._writers[phase] = w
            self._counters[phase] = 0
            self._temp_buffer[phase] = []

    def log_before(self,
                   phase: str,
                   env,
                   action: torch.Tensor,
                   episode: Optional[int]=None):
        """
        Call *before* env.step(action).  Stashes the pre‑step fields.
        """
        B = env.B
        buf = self._temp_buffer[phase]
        buf.clear()
        for i in range(B):
            buf.append({
                'phase': phase,
                'episode': episode,
                'dataset_i': int(env.i[i]),
                'step': int(env.current_step[i]),
                'mask_meta': int(env.mask_meta_window[i, -1]),
                'prices': env.prices[i, -1, :].tolist(),
                'arr_date': str(env.arr_date[i]),
                'arr_year': env.arr_year[i],
                'arr_month': env.arr_month[i],
                'arr_week_of_year': env.arr_week_of_year[i],
                'arr_weeks_in_current_year': env.arr_weeks_in_current_year[i],
                'arr_week_of_month': env.arr_week_of_month[i],
                'arr_weeks_in_current_month': env.arr_weeks_in_current_month[i],
                'arr_months_in_current_year': env.arr_months_in_current_year[i],
                'arr_days_in_week': env.arr_days_in_week[i],
                'arr_days_in_month': env.arr_days_in_month[i],
                'arr_days_in_year': env.arr_days_in_year[i],
                'arr_calendar_days_in_year': env.arr_calendar_days_in_year[i],
                'arr_yr_rank': env.arr_yr_rank[i],
                'arr_et_w': env.arr_et_w[i],
                'arr_et_m': env.arr_et_m[i],
                'arr_et_y': env.arr_et_y[i],
                'w_base': action[i, :env.prices.shape[-1]].tolist(),
                'w_risky': action[i, env.prices.shape[-1]:2*env.prices.shape[-1]].tolist(),
                'weights': env.weights[i].tolist(),
                'amount_risky': float(action[i, 2*env.prices.shape[-1]].item()),
                'theta': float(action[i, 2*env.prices.shape[-1] + 1].item()),
                'pv': float(env.pv[i]),
                'pv_weekly': float(env.week_start_pv[i]),
                'pv_monthly': float(env.month_start_pv[i]),
                'pv_yearly': float(env.year_start_pv[i]),
                'r_week': float(env.r_week[i]),
                'r_month': float(env.r_month[i]),
                'r_year': float(env.r_year[i]),
                'pen_w': float(env.pen_w[i]),
                'pen_m': float(env.pen_m[i]),
                'pen_y': float(env.pen_y[i]),
                'sharpe': float(env.sharpe[i]),
                'sharpe_monthly':  float(env.monthly_sharpes[i][-1] if env.monthly_sharpes[i] else 0.0),
                'sharpe_yearly':   float(env.yearly_sharpes[i][-1]  if env.yearly_sharpes[i]  else 0.0),
                'variance_r': float(env.variance_r[i]),
                'variance_monthly': env.monthly_vars[i][-1] if env.monthly_vars[i] else 0.0,
                'variance_yearly':  env.yearly_vars[i][-1]  if env.yearly_vars[i]  else 0.0,
                'r_env': float(env.r_env[i]),
                'daily_return': float(env.daily_return[i]),
                'lagr_weekly': float(env.lgr_mult_weekly[i]),
                'lagr_monthly': float(env.lgr_mult_monthly[i]),
                'lagr_annual': float(env.lgr_mult_annual[i]),
                # placeholders for after‑step
                #'episode_reward': 0.0,
                'new_pv': None,
                'reward': None,
            })
        # we don’t write anything yet—we wait for log_after()

    def log_after(self, phase: str, env, reward: torch.Tensor, info: dict, episode: Optional[int]=None,):
        """
        Call *after* env.step(action).  Completes each stashed row,
        writes it out immediately, and occasionally flushes.
        """
        B = env.B
        buf = self._temp_buffer[phase]
        writer  = self._writers[phase]
        fhandle = self._files[phase]

        for i in range(B):
            row = buf[i]
            row.update({
                'new_pv': float(env.pv[i]),
                'reward': float(reward[i].item()),
                'episode_rewards': float(env.episode_rewards[i].item()),
                'r_week': float(env.r_week[i]),
                'r_month': float(env.r_month[i]),
                'r_year': float(env.r_year[i]),
                'pen_w': float(env.pen_w[i]),
                'pen_m': float(env.pen_m[i]),
                'pen_y': float(env.pen_y[i]),
                'sharpe': float(env.sharpe[i]),
                'new_weights': env.weights[i].tolist(),
                'sharpe_monthly': float(env.monthly_sharpes[i][-1] if env.monthly_sharpes[i] else 0.0),
                'sharpe_yearly':  float(env.yearly_sharpes[i][-1]  if env.yearly_sharpes[i]  else 0.0),
                'r_env': float(env.r_env[i]),
                'daily_return': float(env.daily_return[i]),
                'lagr_weekly': float(env.lgr_mult_weekly[i]),
                'lagr_monthly': float(env.lgr_mult_monthly[i]),
                'lagr_annual': float(env.lgr_mult_annual[i]),
                'mean_r':         float(env.mean_r[i]),
                'std_r':          float(env.std_r[i]),
                })

            # write the completed record
            self._write_buffer[phase].append(row)
        
        # 2) flush to disk once buffer is big enough
        if len(self._write_buffer[phase]) >= self.write_every:
            for r in self._write_buffer[phase]:
                writer.writerow(r)
            fhandle.flush()
            self._counters[phase] += len(self._write_buffer[phase])
            self._write_buffer[phase].clear()

        # clear temp buffer
        buf.clear()

    def finalize(self, excel_file: str):
        """
        Flush any unwritten buffered rows, close CSVs, and bundle all three
        phases into one XLSX.
        """
        # 1) Flush any remaining buffered rows to CSV
        for phase, buffer in self._write_buffer.items():
            writer = self._writers[phase]
            f      = self._files[phase]
            for row in buffer:
                writer.writerow(row)
            f.flush()
            self._counters[phase] += len(buffer)
            buffer.clear()

        # 2) Close the CSV files
        for f in self._files.values():
            f.close()

        # 3) Create a write-only workbook and stream each CSV into its sheet
        wb = openpyxl.Workbook(write_only=True)
        for phase in ('train', 'validation', 'test'):
            ws = wb.create_sheet(title=phase)
            csv_path = os.path.join(self.log_dir, f"{phase}_logs.csv")
            with open(csv_path, "r", newline="") as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    ws.append(row)

        # 4) Save the XLSX
        wb.save(excel_file)


    def close(self):
        """
        Flush any remaining buffered rows, then close all CSV files.
        """
        # flush leftover buffer
        for phase, buffer in self._write_buffer.items():
            writer = self._writers[phase]
            f      = self._files[phase]
            for row in buffer:
                writer.writerow(row)
            f.flush()
            self._counters[phase] += len(buffer)
            buffer.clear()

        # now close file handles
        for f in self._files.values():
            f.close()

