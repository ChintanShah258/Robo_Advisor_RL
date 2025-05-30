# Writing seperate Data Extractor (than WindowsSampler) as this one gives flexibility to input  
# non-contigous data (i.e. data from different time periods). Reason we chose this way is because 
# since we are getting the Local Targets (Weekly) and Global Targets (Yearly) for our Train, Val, Test 
# data sets and we are getting them from different periods, their local and global targets depend on 
# the data being treated as continous time series (i.e. if training data has years 2015,2016,2017, and 2020)
# then year 2020 should be treated as year 4 instead of year 6. We are splitting time here so that the 
# RL policy will learn on different scenarios. It is not compulsory to split the data into such 
# non-continous frames. Once we get the datasets ready, we will compute the Local and Global Targets.
# The final data will then be merged to the output of 'extract_embeddings' (joined on exact days) as 
# that will form our State Space. We did not perform similar data-splitting technique on 'extract embeddings'
# data because those Embeddings are Market Embeddings and they are dependent on immediate past. Since
# Market Embeddings are indicators of market behaviour, they need to be calculated on sequential time-series
# manner without any data-splitting.

import argparse
import pandas as pd
from typing import List, Tuple, Dict
from dateutil.parser import parse

# ---------------------------
# Helper functions
# ---------------------------
def is_leap_year(y: int) -> bool:
    """Returns True if y is a leap year, otherwise False."""
    return (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0))

def total_weeks_jan1_based(y: int) -> int:
    """
    Returns the total number of weeks in the year if week 1 starts on January 1.
    Any leftover days at the end of the year count as an extra week.
    """
    days_in_year = 366 if is_leap_year(y) else 365
    return (days_in_year - 1) // 7 + 1

def parse_split_args(split_args: List[str]) -> Dict[str, List[Tuple[str,str]]]:
    """
    Parse --split name:start|end name2:start2|end2 ... into
    { name: [(start, end), ...], ... }
    """
    splits: Dict[str, List[Tuple[str,str]]] = {}
    for token in split_args:
        try:
            name, rng = token.split(":", 1)
            start, end = rng.split("|", 1)
        except ValueError:
            raise ValueError(f"Invalid split format '{token}'. Expected name:start|end")
        splits.setdefault(name, []).append((start.strip(), end.strip()))
    return splits

def validate_ranges_against_df(
    df: pd.DataFrame, 
    splits: Dict[str, List[Tuple[str,str]]], 
    date_column: str
):
    min_date = df[date_column].min()
    max_date = df[date_column].max()
    bad = []
    for name, ranges in splits.items():
        for start, end in ranges:
            s = parse(start)
            e = parse(end)
            if s < min_date or e > max_date:
                bad.append((name, start, end))
    if bad:
        print("ERROR: some requested split‐ranges fall outside the data date window.")
        print(f" Data spans from {min_date.date()} to {max_date.date()}.")
        for name, start, end in bad:
            print(f"  • {name}: {start} → {end}")
        exit(1)

def split_by_date_ranges(
    df: pd.DataFrame,
    date_ranges: List[Tuple[str,str]],
    date_column: str = "Dates"
) -> pd.DataFrame:
    # (unchanged)
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
    parts = []
    for start, end in date_ranges:
        s = pd.to_datetime(parse(start))
        e = pd.to_datetime(parse(end))
        mask = (df[date_column] >= s) & (df[date_column] <= e)
        sub = df.loc[mask]
        if not sub.empty:
            parts.append(sub)
    if not parts:
        return df.iloc[0:0]  # empty with same columns
    out = pd.concat(parts, ignore_index=True).sort_values(date_column)
    return out

# def create_dataset_from_ranges(
#     df: pd.DataFrame,
#     date_ranges_dict: dict,
#     date_column: str = 'Dates'
# ) -> dict:
#     """
#     Create a dictionary of datasets by splitting the DataFrame into different date ranges.
#     """
#     result = {}
#     for dataset_name, ranges in date_ranges_dict.items():
#         result[dataset_name] = split_by_date_ranges(df, ranges, date_column)
#     return result

# ---------------------------
# DataProcessor class
# ---------------------------
class DataProcessor:
    """
    DataProcessor loads stock data from a CSV file, splits the data into multiple sets based on 
    provided date ranges, and computes reward target columns (annual, monthly, weekly, daily) as well 
    as various date-related metrics.
    """
    def __init__(self, raw_df: pd.DataFrame, embed_df: pd.DataFrame, asset_list: List[str],date_column="Dates", annual_target=0.03,
                 hist_window=10):
        
        # 1) remember which assets
        self.asset_list = asset_list
        self.date_column = date_column
                
        # 2) build the exact column-names we need
        price_cols   = [f"{sym}"    for sym in asset_list]
        volume_cols  = [f"{sym}_Volume"   for sym in asset_list]
        vol10d_cols  = [f"{sym}_Vol_10D"   for sym in asset_list]
        vol30d_cols  = [f"{sym}_Vol_30D"   for sym in asset_list]
        vol90d_cols  = [f"{sym}_Vol_90D"   for sym in asset_list]
        
        price_log_sd_cols = [f"{sym}_log_sd" for sym in asset_list]
        volume_log_sd_cols = [f"{sym}_Volume_log_sd" for sym in asset_list]
        vol10d_log_sd_cols = [f"{sym}_Vol_10D_log_sd" for sym in asset_list]
        vol30d_log_sd_cols = [f"{sym}_Vol_30D_log_sd" for sym in asset_list]
        vol90d_log_sd_cols = [f"{sym}_Vol_90D_log_sd" for sym in asset_list]
        
        #We can either take raw prices or standardized(log(prices)) as input
        #We are using raw prices to compute returns while using standardized(log(prices)) to 
        #act as state space (and thus get the actions). The standardized(log(prices)) have unit variance
        #and are good for NN computations
        needed = [date_column] + price_cols + price_log_sd_cols + volume_log_sd_cols + \
        vol10d_log_sd_cols + vol30d_log_sd_cols + vol90d_log_sd_cols        
        
        self.raw_df = raw_df[needed].copy()
        self.embed_df = embed_df
        self.annual_target = annual_target
        self.hist_window   = hist_window
        self.merged = pd.merge(self.raw_df, embed_df, on=date_column, how="inner")
        # record column partitions
        self.raw_cols = [c for c in self.raw_df.columns if c != self.date_column]
        self.embed_cols = [c for c in embed_df.columns if c != self.date_column]
        self.datasets = {}

    def split_data(self, splits: Dict[str, List[Tuple[str,str]]]):
        for name, ranges in splits.items():
            self.datasets[name] = split_by_date_ranges(self.merged, ranges, self.date_column)

    # def load_data(self) -> pd.DataFrame:
    #     """Loads the CSV data and converts the date column."""
    #     self.df = pd.read_csv(self.file_path)
    #     if self.date_column not in self.df.columns:
    #         raise ValueError(f"Date column '{self.date_column}' not found in data!")
    #     self.df[self.date_column] = pd.to_datetime(self.df[self.date_column])
    #     return self.df

    # def split_data(self, date_ranges_dict: dict) -> dict:
    #     """
    #     Splits the loaded data according to provided date ranges.
    #     Expects date_ranges_dict to be a dictionary with keys (e.g., 'train', 'test', 'validation')
    #     and values as lists of (start_date, end_date) tuples.
    #     """
    #     if self.df is None:
    #         self.load_data()
    #     self.datasets = create_dataset_from_ranges(self.df, date_ranges_dict, date_column=self.date_column)
    #     return self.datasets

    def add_rewards(self) -> dict:
        """
        Adds computed reward target columns and other date-based metrics to each dataset in self.datasets.
        The metrics include: year, month, week of month, week of year, days in week/month/year, and various
        reward target columns (annual, monthly, weekly, daily and their cumulative versions).
        """
        # if self.datasets is None:
        #     raise ValueError("Data not split yet. Call split_data() first.")
        
        updated = {}
        for name, df in self.datasets.items():
            # 1) copy + reset index
            d = df.copy()
            
            # 2) split warm vs work
            warm = d.iloc[: self.hist_window - 1].copy()
            work = d.iloc[self.hist_window :].copy().reset_index(drop=True)
            # Ensure date column is datetime
            
            # 3) redirect computations to `d = work`
            d = work
            
            d[self.date_column] = pd.to_datetime(d[self.date_column])

            # Extract year and month
            d['year'] = d[self.date_column].dt.year
            d['month'] = d[self.date_column].dt.month

            # Create mapping for year_rank
            unique_years = sorted(d['year'].unique())
            year_map = {yr: idx + 1 for idx, yr in enumerate(unique_years)}
            d['year_rank'] = d['year'].map(year_map)

            # Total unique months in current year
            d['months_in_current_year'] = d.groupby('year')['month'].transform(lambda x: x.nunique())

            # Week of month: simple 7-day blocks starting from the 1st
            d['week_of_month'] = (d[self.date_column].dt.day - 1) // 7 + 1

            # Total weeks in current month
            d['weeks_in_current_month'] = d.groupby(['year', 'month'])['week_of_month'].transform(lambda x: x.nunique())

            # Week of year: custom calculation where week 1 starts on January 1.
            d['week_of_year'] = d[self.date_column].apply(
                lambda dt: ((dt - pd.Timestamp(dt.year, 1, 1)).days // 7) + 1
            )

            # Total weeks in current year
            d['weeks_in_current_year'] = d.groupby('year')['week_of_year'].transform(lambda x: x.nunique())

            # Maximum possible weeks in the year
            d['max_weeks_in_year'] = d['year'].apply(total_weeks_jan1_based)

            # Days in week, month, and year (from available data)
            d['days_in_week'] = d.groupby(['year', 'month', 'week_of_month'])[self.date_column].transform('size')
            d['days_in_month'] = d.groupby(['year', 'month'])[self.date_column].transform('size')
            d['days_in_year'] = d.groupby('year')[self.date_column].transform('size')

            # Calendar days in the year
            d['calendar_days_in_year'] = d['year'].apply(lambda y: 366 if is_leap_year(y) else 365)

            # Compute various reward target columns
            d['annual_target_(daily)'] = self.annual_target * d['days_in_year'] / 252
            d['annual_target_(daily)_cumulative'] = (
            d['annual_target_(daily)'].div(d['days_in_year']).groupby(d['year']).cumsum())
            d['annual_target_(weekly)'] = (self.annual_target * d['weeks_in_current_year']) / d['max_weeks_in_year']
            d['annual_target_(weekly)_cumulative'] = (
                d.assign(weekly_fraction=lambda df2: df2['annual_target_(weekly)'] / df2['weeks_in_current_year'])
                .drop_duplicates(['year', 'week_of_year'])
                .groupby('year')['weekly_fraction'].cumsum()
                .reindex(d.index, method='ffill')
            )
            d['annual_target_(monthly)'] = (self.annual_target * d['months_in_current_year']) / 12
            d['annual_target_(monthly)_cumulative'] = (
                d.assign(monthly_fraction=lambda df2: df2['annual_target_(monthly)'] / df2['months_in_current_year'])
                .drop_duplicates(['year', 'month'])
                .groupby('year')['monthly_fraction'].cumsum()
                .reindex(d.index, method='ffill')
            )
            d['monthly_target_(daily)'] = (
                d['days_in_month'] * (self.annual_target * d['months_in_current_year'] / 12)
            ) / d['days_in_year']
            d['monthly_target_(daily)_cumulative'] = (
                d['monthly_target_(daily)'].div(d['days_in_month']).groupby([d['year'], d['month']]).cumsum())
            d['monthly_target_(weekly)'] = (
                d['weeks_in_current_month'] * (self.annual_target * d['months_in_current_year'] / 12)
            ) / d['weeks_in_current_year']
            d['monthly_target_(weekly)_cumulative'] = (
                d.assign(weekly_fraction=lambda df2: df2['monthly_target_(weekly)'] / df2['weeks_in_current_month'])
                .drop_duplicates(['year', 'month', 'week_of_month'])
                .groupby([d['year'], d['month']])['weekly_fraction'].cumsum()
                .reindex(d.index, method='ffill')
            )
            d['weekly_target_(daily)'] = (
                d['days_in_week'] * (self.annual_target * d['weeks_in_current_year'] / d['max_weeks_in_year'])
            ) / d['days_in_year']
            d['weekly_target_(daily)_cumulative'] = (
                d['weekly_target_(daily)'].div(d['days_in_week'])
                .groupby([d['year'], d['month'], d['week_of_month']]).cumsum()
            )
            
            # Daily target based on annual target
            d['daily_target_(annual)'] = d['annual_target_(daily)'] / d['days_in_year']
            
            # Cumulative daily target based on annual target
            d['daily_target_(annual)_cumulative'] = (
                d['daily_target_(annual)'].div(d['days_in_year'])
                .groupby(d['year']).cumsum()
                .reindex(d.index, method='ffill')
            )
            
            # Daily target based on Monthly target
            d['daily_target_(monthly)'] = d['monthly_target_(daily)'] / d['days_in_month']
            
            # Cumulative daily target based on annual target
            d['daily_target_(monthly)_cumulative'] = (
                d['daily_target_(monthly)'].div(d['days_in_month'])
                .groupby([d['year'],d['month']]).cumsum()
                .reindex(d.index, method='ffill')
            )
            
            # Daily target based on Monthly target
            d['daily_target_(weekly)'] = d['weekly_target_(daily)'] / d['days_in_week']
            
            # Cumulative daily target based on annual target
            d['daily_target_(weekly)_cumulative'] = (
                d['daily_target_(weekly)'].div(d['days_in_week'])
                .groupby([d['year'],d['month'],d['week_of_month']]).cumsum()
                .reindex(d.index, method='ffill')
            )
            
            # 4) identify new columns                        
            # Now pull out reward column names from `d`
            reward_cols = [
                c for c in d.columns
                if c not in ([self.date_column] + self.raw_cols + self.embed_cols)
            ]

            # 5) zero‑pad warm for those
            for c in reward_cols:
                warm[c] = 0
            
            # 6) optional mask
            warm['mask_meta'] = 0
            work['mask_meta'] = 1
            
            # 7) recombine + reorder
            combined = pd.concat([warm, work], ignore_index=True)
            ordered  = (
                ['mask_meta']
                + [self.date_column]
                + self.raw_cols
                + reward_cols
                + self.embed_cols
            )
            updated[name] = combined[ordered]

            # replace old datasets
            self.datasets = updated

    def process_all(self, date_ranges_dict: dict) -> dict:
        """
        Full pipeline: load data, split by date ranges, and add reward-related columns.
        Returns a dictionary of processed DataFrames.
        """
        self.load_data()
        self.split_data(date_ranges_dict)
        self.add_rewards()
        return self.datasets

# # ---------------------------
# # Main block with argparse and merging of selected columns
# # ---------------------------
# def parse_date_ranges(date_range_list: List[str]) -> List[Tuple[str, str]]:
#     ranges = []
#     for dr in date_range_list:
#         # Use '|' as the splitter.
#         parts = dr.split("|", 1)
#         if len(parts) != 2:
#             raise ValueError(f"Invalid date range format: {dr}. Expected format: 'start_date|end_date'.")
#         start, end = parts
#         ranges.append((start.strip(), end.strip()))
#     return ranges


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw_file", required=True, help="CSV with Dates + raw market data")
    p.add_argument("--embed_file", required=True, help="CSV with Dates + embeddings")
    p.add_argument("--annual_target", type=float, default=0.03)
    p.add_argument(
    "--asset_list",
    nargs="+",
    required=True,
    help="List of asset symbols, e.g. --asset_list ECL NEM APD FCX VMC MLM"
    )
    p.add_argument("--hist_window", type=int, default=10, required = True, help="days of historical data to use")
    p.add_argument(
        "--split",
        nargs="+",
        required=True,
        help="One or more splits: name:start|end  e.g. train:2015-01-01|2017-12-31 test:2018-01-01|2019-12-31"
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,                # change default to None
        help="Path to the output Excel file.  If omitted, will be auto-named from asset_list."
    )
    args = p.parse_args()
    
    # ─── auto-generate output name if user didn’t supply one ─────────────────
    if args.output is None:
        # take at most the last 4 symbols
        suffix_assets = args.asset_list[-4:]
        # join with underscores
        suffix = "_".join(suffix_assets)
        args.output = f"{suffix}_final_input_data.xlsx"

    # load
    raw = pd.read_csv(args.raw_file, parse_dates=["Dates"])
    embed = pd.read_csv(args.embed_file, parse_dates=["Dates"])
    # parse & validate
    splits = parse_split_args(args.split)
    #validate_ranges_against_df(raw, splits, "Dates")

    # process
    proc = DataProcessor(raw, embed, annual_target=args.annual_target,
                         asset_list=args.asset_list,hist_window=args.hist_window)
    proc.split_data(splits)
    proc.add_rewards()

    # write out each split to its own sheet
    with pd.ExcelWriter(args.output) as w:
        for name, df in proc.datasets.items():
            df.to_excel(w, sheet_name=name, index=False)
    print("Wrote splits:", ", ".join(proc.datasets.keys()))


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Process reward targets for stock data with user-defined parameters."
#     )
#     parser.add_argument("--file_path", type=str, required=True, help="Path to CSV file.")
#     parser.add_argument("--date_column", type=str, default="Dates", help="Name of the date column in the CSV.")
#     parser.add_argument("--annual_target", type=float, default=0.03, help="Annual target return (e.g., 0.03 for 3%).")
#     parser.add_argument("--global_target", type=str, choices=["annual", "monthly", "weekly"], default="annual",
#                         help="Which global reward target structure to use.")
#     parser.add_argument("--local_target", type=str, choices=["daily", "weekly", "monthly"], default="daily",
#                         help="Which local reward target structure to use.")
#     # Date ranges for each split as multiple arguments in the form "start_date,end_date"
#     parser.add_argument("--train_dates", type=str, nargs="+", required=True,
#                         help='Train date ranges, each as "start_date,end_date".')
#     parser.add_argument("--test_dates", type=str, nargs="+", required=True,
#                         help='Test date ranges, each as "start_date,end_date".')
#     parser.add_argument("--validation_dates", type=str, nargs="+", required=True,
#                         help='Validation date ranges, each as "start_date,end_date".')
#     parser.add_argument("--output", type=str, default="final_input_data.xlsx", help="Output Excel file name.")

#     args = parser.parse_args()

#     date_ranges_dict = {
#         "train": parse_date_ranges(args.train_dates),
#         "test": parse_date_ranges(args.test_dates),
#         "validation": parse_date_ranges(args.validation_dates)
#     }

#     # Instantiate DataProcessor and process data
#     processor = DataProcessor(file_path=args.file_path, date_column=args.date_column, annual_target=args.annual_target)
#     processed_datasets = processor.process_all(date_ranges_dict)

#     # Define mapping for valid global/local combinations:
#     # global_target = {global_target}_target_{local_target}
#     # local_target  = {global_target}_target_{local_target}_cumulative
#     global_local_to_columns = {
#         ('annual', 'monthly'): ("annual_target_(monthly)", "annual_target_(monthly)_cumulative"),
#         ('annual', 'weekly'):  ("annual_target_(weekly)",  "annual_target_(weekly)_cumulative"),
#         ('annual', 'daily'):   ("annual_target_(daily)",   "annual_target_(daily)_cumulative"),
#         ('monthly', 'weekly'): ("monthly_target_(weekly)", "monthly_target_(weekly)_cumulative"),
#         ('monthly', 'daily'):  ("monthly_target_(daily)",  "monthly_target_(daily)_cumulative"),
#         ('weekly', 'daily'):   ("weekly_target_(daily)",   "weekly_target_(daily)_cumulative"),
#     }

#     combo = (args.global_target, args.local_target)
#     if combo not in global_local_to_columns:
#         raise ValueError(f"Invalid combination: global_target={args.global_target}, local_target={args.local_target}")
#     global_col, local_col = global_local_to_columns[combo]
#     print(f"Selected columns based on arguments: Global = '{global_col}', Local = '{local_col}'")

#     # Merge the selected reward columns with the original input DataFrame on the date column.
#     # Here we use the "train" split's computed reward columns as an example.
#     reward_df = processed_datasets.get("train")
#     if reward_df is None:
#         reward_df = processor.df  # Fallback to original if "train" is not available

#     # Ensure that the reward_df has the date column
#     reward_df = reward_df[[args.date_column, global_col, local_col]].drop_duplicates(args.date_column)
#     merged_df = pd.merge(
#         processor.df,
#         reward_df,
#         on=args.date_column,
#         how="inner"
#     )

#     # Write the merged DataFrame (with selected reward columns) and each split to an Excel file.
#     with pd.ExcelWriter(args.output) as writer:
#         merged_df.to_excel(writer, sheet_name="merged", index=False)
#         for sheet_name, df in processed_datasets.items():
#             df.to_excel(writer, sheet_name=sheet_name, index=False)

#     # Print summary information
#     print("Merged DataFrame shape:", merged_df.shape)
#     print("Merged DataFrame columns:", merged_df.columns.tolist())
