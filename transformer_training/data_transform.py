import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('/home/student/robo_advisor_new/transformer_training/data_sp_500/SP500_Final.csv')
# convert the Dates column into real Timestamps
df['Dates'] = pd.to_datetime(df['Dates'], infer_datetime_format=True)

n_assets = 100
df_2000_2014 = df[(df['Dates'] >= '01-01-2000') & (df['Dates'] <= '12-31-2014')]

df_dates_2014 = df_2000_2014['Dates']
df_price_2014 = df_2000_2014.iloc[:, 1:n_assets + 1]
df_volume_2014 = df_2000_2014.iloc[:, n_assets + 1:n_assets + 101]
df_vol_10d_2014 = df_2000_2014.iloc[:, n_assets + 101:n_assets + 201]
df_vol_30d_2014 = df_2000_2014.iloc[:, n_assets + 201:n_assets + 301]
df_vol_90d_2014 = df_2000_2014.iloc[:, n_assets + 301:n_assets + 401]

df_log_price_returns_2014 = np.log(df_price_2014).diff().fillna(0)
df_log_volume_2014 = np.log1p(df_volume_2014)
df_log_vol_10d_2014 = np.log(df_vol_10d_2014).fillna(0)
df_log_vol_30d_2014 = np.log(df_vol_30d_2014).fillna(0)
df_log_vol_90d_2014 = np.log(df_vol_90d_2014).fillna(0)

# concat all three pieces back into one DataFrame
df_all_2014 = pd.concat([df_log_price_returns_2014, df_log_volume_2014, df_log_vol_10d_2014, \
                    df_log_vol_30d_2014,df_log_vol_90d_2014], axis=1)

scaler = StandardScaler()
df_scaled_2014 = pd.DataFrame(
    scaler.fit_transform(df_all_2014), 
    index=df_all_2014.index, 
    columns=df_all_2014.columns
)

df_pretrain_transformer = pd.concat([df_dates_2014, df_scaled_2014], axis=1)
df_pretrain_transformer.to_csv('/home/student/robo_advisor_new/transformer_training/data_sp_500/SP500_Pretrain_Transformer.csv', index=False)

# save scaler for inference on 2015-2020
joblib.dump(scaler, "pretrain_scaler.pkl")

# load scaler
scaler = joblib.load("pretrain_scaler.pkl")

df_2015_2024 = df[(df['Dates'] >= '01-01-2015') & (df['Dates'] <= '12-31-2024')]

df_2015_2024.to_csv('/home/student/robo_advisor_new/base_td3/data_prep/SP500_2015_2024.csv', index=False)

df_dates_2015 = df_2015_2024['Dates']
df_price_2015 = df_2015_2024.iloc[:, 1:n_assets + 1]
df_volume_2015 = df_2015_2024.iloc[:, n_assets + 1:n_assets + 101]
df_vol_10d_2015 = df_2015_2024.iloc[:, n_assets + 101:n_assets + 201]
df_vol_30d_2015 = df_2015_2024.iloc[:, n_assets + 201:n_assets + 301]
df_vol_90d_2015 = df_2015_2024.iloc[:, n_assets + 301:n_assets + 401]

df_log_price_returns_2015 = np.log(df_price_2015).diff().fillna(0)
df_log_volume_2015 = np.log1p(df_volume_2015)
df_log_vol_10d_2015 = np.log(df_vol_10d_2015).fillna(0)
df_log_vol_30d_2015 = np.log(df_vol_30d_2015).fillna(0)
df_log_vol_90d_2015 = np.log(df_vol_90d_2015).fillna(0)

# concat all three pieces back into one DataFrame
df_all_2015 = pd.concat([df_log_price_returns_2015, df_log_volume_2015, df_log_vol_10d_2015, \
                    df_log_vol_30d_2015,df_log_vol_90d_2015], axis=1)

df_scaled_2015 = pd.DataFrame(
    scaler.transform(df_all_2015),           # use transform, not fit_transform
    index=df_all_2015.index,
    columns=df_all_2015.columns
)

# 2) re-attach the Dates column
df_embed_2015 = pd.concat([df_dates_2015, df_scaled_2015], axis=1)
df_embed_2015.to_csv('/home/student/robo_advisor_new/transformer_training/data_sp_500/SP500_Transformer_Embeddings.csv', index=False)


#-------------- Getting Raw and Log Returns for 2015-2024 ----------------#
df_2015_2024_log_sd = pd.read_csv('/home/student/robo_advisor_new/base_td3/data_prep/SP500_2015_2024.csv')
df_2015_2024_log_sd = df_2015_2024_log_sd.rename(columns=lambda c: f"{c}_log_sd" if c != "Dates" else c)
df_dates_2015 = df_2015_2024_log_sd['Dates']
df_price_2015 = df_2015_2024_log_sd.iloc[:, 1:n_assets + 1]
df_volume_2015 = df_2015_2024_log_sd.iloc[:, n_assets + 1:n_assets + 101]
df_vol_10d_2015 = df_2015_2024_log_sd.iloc[:, n_assets + 101:n_assets + 201]
df_vol_30d_2015 = df_2015_2024_log_sd.iloc[:, n_assets + 201:n_assets + 301]
df_vol_90d_2015 = df_2015_2024_log_sd.iloc[:, n_assets + 301:n_assets + 401]

df_log_price_returns_2015 = np.log(df_price_2015).diff().fillna(0)
df_log_volume_2015 = np.log1p(df_volume_2015)
df_log_vol_10d_2015 = np.log(df_vol_10d_2015).fillna(0)
df_log_vol_30d_2015 = np.log(df_vol_30d_2015).fillna(0)
df_log_vol_90d_2015 = np.log(df_vol_90d_2015).fillna(0)

# concat all three pieces back into one DataFrame
df_all_2015 = pd.concat([df_log_price_returns_2015, df_log_volume_2015, df_log_vol_10d_2015, \
                    df_log_vol_30d_2015,df_log_vol_90d_2015], axis=1)

scaler_log_sd = StandardScaler()
df_scaled_2015_log_sd = pd.DataFrame(
    scaler_log_sd.fit_transform(df_all_2015), 
    index=df_all_2015.index, 
    columns=df_all_2015.columns
)
df_scaled_2015_log_sd = pd.concat([df_dates_2015, df_scaled_2015_log_sd], axis=1)
df_scaled_2015_log_sd.to_csv('/home/student/robo_advisor_new/base_td3/data_prep/SP500_2015_2024_log_sd.csv', index=False)

# ensure Dates is datetime in the scaled‐log‐sd frame
df_scaled_2015_log_sd['Dates'] = pd.to_datetime(df_scaled_2015_log_sd['Dates'])

df_raw_log_sd = pd.merge(df_2015_2024, df_scaled_2015_log_sd, on='Dates', how='inner')
df_raw_log_sd.to_csv('/home/student/robo_advisor_new/base_td3/data_prep/SP500_2015_2024_raw_log_sd.csv', index=False)
