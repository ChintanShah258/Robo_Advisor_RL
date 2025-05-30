import sys, os
project_root = os.path.abspath(os.path.join(__file__, '..', '..'))
sys.path.insert(0, project_root)

import torch
import pandas as pd
import argparse
import os
from transformer_training.transformer import MASTER  # Ensure this correctly points to your MASTER model definition

def parse_ranges(s: str):
    out = []
    for part in s.split(','):
        lo, hi = part.split('-')
        out.append((int(lo), int(hi)))
    return out

def load_model(model_path, input_feature_ranges, gate_input_ranges, d_feat, d_model, t_nhead, 
               s_nhead, T_dropout_rate, S_dropout_rate, beta):
    """
    Load the pre-trained MASTER model with saved weights.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MASTER(
        input_feature_ranges = input_feature_ranges,
        gate_input_ranges    = gate_input_ranges,
        d_feat=d_feat,
        d_model=d_model,
        t_nhead=t_nhead,
        s_nhead=s_nhead,
        T_dropout_rate=T_dropout_rate,
        S_dropout_rate=S_dropout_rate,
        beta=beta,
        aggregate_output=True  # Aggregating per window for RL model
    ).to(device)
    
    # load the checkpoint dict
    ckpt = torch.load(model_path, map_location=device)
    # pull out only the MASTER weights
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, device

def extract_embeddings(model, device, data, dates, window_size=50):
    """
    Pass unseen data through the MASTER model to get embeddings.
    
    Args:
        model: Pretrained MASTER model.
        device: 'cuda' or 'cpu'
        data: DataFrame of shape (num_days, num_stocks)
        dates: List of dates corresponding to each row in data.
        window_size: Number of days per input window.

    Returns:
        embeddings_df: DataFrame containing dates and embeddings.
    """
    embeddings = []
    valid_dates = []  # Store the dates corresponding to each embedding
    num_days = len(data)

    with torch.no_grad():  # No gradients needed during inference
        for start in range(num_days - window_size):
            window_data = torch.tensor(data[start: start + window_size].values, dtype=torch.float32).unsqueeze(0).to(device)
            embedding = model(window_data)  # Shape: [1, d_model]
            embeddings.append(embedding.cpu().numpy().flatten())  # Convert to 1D array
            valid_dates.append(dates[start + window_size])  # Date corresponding to the next day's decision
    
    embeddings_df = pd.DataFrame(embeddings)
    embeddings_df.insert(0, "Dates", valid_dates)  # Insert dates as first column
    
    return embeddings_df

def main(args):
    # Load data
    df = pd.read_csv(args.unseen_data_path)
    
    # Extract dates column
    if "Dates" in df.columns:
        dates = df["Dates"].tolist()
        df = df.drop(columns=["Dates"])  # Remove dates column from feature data
        stock_tickers = df.columns.tolist()  # Extract stock tickers
        #print(stock_tickers)
    else:
        raise ValueError("CSV file must have a 'dates' column with timestamps!")

    input_ranges = parse_ranges(args.input_feature_ranges)
    gate_ranges  = parse_ranges(args.gate_input_ranges)
    d_feat = sum((hi - lo) for lo, hi in input_ranges)
    
    # Load pre-trained MASTER model
    model, device = load_model(
        model_path=args.model_path,
        input_feature_ranges = input_ranges,
        gate_input_ranges    = gate_ranges,
        d_feat               = d_feat,
        d_model=args.d_model,
        t_nhead=args.t_nhead,
        s_nhead=args.s_nhead,
        T_dropout_rate=args.T_dropout_rate,
        S_dropout_rate=args.S_dropout_rate,
        beta=args.beta,
        )
    
    # Extract embeddings with dates
    embeddings_df = extract_embeddings(model, device, df, dates, window_size=args.window_size)
    
    # Construct output filename dynamically based on stock tickers
    #stock_ticker_str = "_".join(stock_tickers[:min(10, len(stock_tickers))])  # Limit to first 5 tickers to avoid long filenames
    output_filename = f"{args.save_prefix}_sp500_final.csv"
    output_path = os.path.join(args.output_dir, output_filename)
    embeddings_df.to_csv(output_path, index=False)
    
    print(f"Extracted embeddings saved to {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract embeddings from unseen stock price data using pretrained MASTER model"
    )

    parser.add_argument('--unseen_data_path', type=str, required=True,
                        help='Path to the CSV file with unseen stock price data.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the pretrained MASTER model file.')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save extracted embeddings.')
    parser.add_argument('--save_prefix', type=str, default='extracted',
                        help='Prefix for the saved embedding file.')

    # Model hyperparameters (must match pretraining setup)
    parser.add_argument('--d_model', type=int, default=128,
                        help='Dimension of the model after projection.')
    parser.add_argument('--t_nhead', type=int, default=1,
                        help='Number of attention heads for temporal attention.')
    parser.add_argument('--s_nhead', type=int, default=2,
                        help='Number of attention heads for spatial attention.')
    parser.add_argument('--T_dropout_rate', type=float, default=0.5,
                        help='Dropout rate for temporal attention.')
    parser.add_argument('--S_dropout_rate', type=float, default=0.5,
                        help='Dropout rate for spatial attention.')
    parser.add_argument('--input_feature_ranges', type=str, required=True,
                        help='e.g. "0-9,31-35"')
    parser.add_argument('--gate_input_ranges',    type=str, required=True,
                        help='e.g. "10-20,25-30"')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Beta value (temperature) for the gate mechanism.')
    parser.add_argument('--window_size', type=int, default=50,
                        help='Number of days per window (T).')
    parser.add_argument('--aggregate_output', type=str, default='True',
                        help='Average the output over window.')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)

