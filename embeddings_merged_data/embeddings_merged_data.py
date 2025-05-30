import pandas as pd
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial_data_file_path', required=True, help='Path to the input Excel file with raw data.')
    parser.add_argument('--embeddings_file_path', required=True, help='Path to the CSV file containing embeddings data.')
    parser.add_argument('--output', required=True, help='Path for the output CSV file.')
    parser.add_argument('--date_column', default='Dates', help='Name of the date column in both files.')
    return parser.parse_args() 

def final_data(df: pd.DataFrame, df_embeddings: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    1. Strips column names in df to remove any trailing/leading spaces.
    2. Checks if date_column exists in df.
    3. Converts the date_column to datetime, extracts only the date portion.
    4. Merges (left join) df with df_embeddings on date_column.
    """
    # Ensure consistent column naming (strip whitespace)
    df.columns = df.columns.str.strip()

    # Check if the date column exists in the rewards dataframe
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found in the rewards dataframe.")

    # Convert the date column to datetime (date only)
    df[date_column] = pd.to_datetime(df[date_column]).dt.date

    # Merge the rewards dataframe with embeddings using a inner join
    merged_df = pd.merge(df, df_embeddings, on=date_column, how='inner')
    return merged_df

def main(args):
    # Read all sheets from the Excel file into a dictionary (keys are sheet names)
    df_raw = pd.read_csv(args.initial_data_file_path)
    df_raw.columns = df_raw.columns.str.strip()

    # Read the embeddings CSV file
    df_embeddings = pd.read_csv(args.embeddings_file_path)
    # Clean column names in the embeddings dataframe
    df_embeddings.columns = df_embeddings.columns.str.strip()

    # Ensure date column exists in both
    for df_name, df in [('raw', df_raw), ('embeddings', df_embeddings)]:
        if args.date_column not in df.columns:
            raise ValueError(f"Date column '{args.date_column}' not found in the {df_name} data.")

    # Normalize date column to date-only
    df_raw[args.date_column]        = pd.to_datetime(df_raw[args.date_column]).dt.date
    df_embeddings[args.date_column] = pd.to_datetime(df_embeddings[args.date_column]).dt.date

    # Merge raw + embeddings: raw columns first, then embeddings columns
    merged_df = pd.merge(df_raw,df_embeddings,on=args.date_column,how='inner'
        #suffixes=('', '_emb')  # avoid collisions
    )

    # Write to output CSV
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    merged_df.to_csv(args.output, index=False)

if __name__ == '__main__':
    args = parse_args()
    main(args)
