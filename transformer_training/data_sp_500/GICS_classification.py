import time
import pandas as pd
import yfinance as yf

# 1) Read your tickers from CSV
df_input = pd.read_csv('Ticker_list.csv')
tickers = df_input['Ticker'].dropna().astype(str).tolist()

# 2) Prepare to collect results
results = []

# 3) Loop with retry + throttling
for ticker in tickers:
    sector = industry = long_name = market_cap = None

    for attempt in range(1, 4):  # up to 3 tries
        try:
            info = yf.Ticker(ticker).info
            sector     = info.get("sector",     None)
            industry   = info.get("industry",   None)
            long_name  = info.get("longName",   None)
            market_cap = info.get("marketCap",  None)
            break
        except Exception as e:
            print(f"⚠️ {ticker} lookup failed on attempt {attempt}: {e}")
            time.sleep(attempt * 1.0)  # back‑off: 1s, 2s, 3s

    # Append even if some fields are None—will drop later
    results.append({
        "Ticker":     ticker,
        "Sector":     sector,
        "Industry":   industry,
        "Long Name":  long_name,
        "Market Cap": market_cap
    })

    # throttle overall fetch rate
    time.sleep(0.5)

# 4) Build DataFrame
df = pd.DataFrame(results)

# 5) Drop any row with a missing field
df.dropna(inplace=True)

# 6) Collapse Consumer sectors into a single "Consumer"
df['Sector'] = df['Sector'].replace({
    'Consumer Cyclical': 'Consumer',
    'Consumer Defensive': 'Consumer'
})

# 7) Sort by Sector (asc) then Market Cap (desc within each sector)
df.sort_values(
    by=['Sector', 'Market Cap'],
    ascending=[True, False],
    inplace=True
)

# 8) Write out cleaned, sorted CSV
df.to_csv("sp500_gics_classification_with_mcap_cleaned.csv", index=False)

print("Done — wrote sp500_gics_classification_with_mcap_cleaned.csv")
