import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import glob

path_pattern = '2024/openaq_location_783911_measurments*.csv'
csv_files = glob.glob(path_pattern)
df_list = []
for file in csv_files:
    df = pd.read_csv(file)
    df_list.append(df)
df_combined = pd.concat(df_list, ignore_index=True)
df_combined['datetimeUtc'] = pd.to_datetime(df_combined['datetimeUtc']).dt.tz_localize(None)

#TODO: refine the prediction model

df_prophet = df_combined[['datetimeUtc', 'value']].rename(columns={'datetimeUtc': 'ds', 'value': 'y'})
m = Prophet(changepoint_prior_scale=0.01)
m.fit(df_prophet)
future = m.make_future_dataframe(periods=300, freq='h')
forecast = m.predict(future)
fig = m.plot(forecast)
plt.show()
