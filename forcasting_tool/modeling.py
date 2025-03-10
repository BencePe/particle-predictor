import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import holidays
from datetime import timedelta
import data_processing

def get_holidays(year):
    hu_holidays = holidays.Hungary(years=[year])
    holiday_list = []
    for date, name in hu_holidays.items():
        holiday_list.append({
            'holiday': name,
            'ds': pd.to_datetime(date),
            'lower_window': 0,
            'upper_window': 1
        })
    return pd.DataFrame(holiday_list)

def build_yearly_model(df, year, start_date, end_date):
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    df_year = df[(df['ds'] >= start_dt) & (df['ds'] <= end_dt) & (df['ds'].dt.year == year)].copy()
    
    if df_year.empty:
        print(f"No data available for {year} in the specified window.")
        return None, None

    df_year['month'] = df_year['ds'].dt.month
    holiday_df = get_holidays(year)
    
    m = Prophet(
        changepoint_prior_scale=0.01,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=holiday_df
    )
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    m.add_regressor('month')
    m.fit(df_year)
    return m, df_year

def forecast_future_period(m, forecast_start, forecast_end, freq='h'):
    training_end = m.history['ds'].max()
    forecast_end_dt = pd.to_datetime(forecast_end)
    additional_hours = int((forecast_end_dt - training_end) / timedelta(hours=1))
    
    future = m.make_future_dataframe(periods=additional_hours, freq=freq)
    future['month'] = future['ds'].dt.month
    forecast_full = m.predict(future)
    
    forecast_period = forecast_full[(forecast_full['ds'] >= pd.to_datetime(forecast_start)) &
                                    (forecast_full['ds'] <= pd.to_datetime(forecast_end))]
    return forecast_period, forecast_full

def predict_future_parallel(clean_csv, forecast_start, forecast_end):
    clean_df = pd.read_csv(clean_csv)
    if 'datetimeUtc' not in clean_df.columns or 'value' not in clean_df.columns:
        print("Error: Required columns missing in cleaned_data.csv")
        return
    
    df_prophet = clean_df[['datetimeUtc', 'value']].rename(columns={'datetimeUtc': 'ds', 'value': 'y'})
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], errors='coerce')
    
    m_2023, train_2023 = build_yearly_model(df_prophet, 2023, "2023-03-31", "2023-06-27")
    m_2024, train_2024 = build_yearly_model(df_prophet, 2024, "2024-02-14", "2024-06-27")
    
    if m_2023 is None or m_2024 is None:
        print("One or both models failed to train.")
        return
    
    fc_period_2023, fc_full_2023 = forecast_future_period(m_2023, forecast_start, forecast_end)
    fc_period_2024, fc_full_2024 = forecast_future_period(m_2024, forecast_start, forecast_end)
    
    forecast_combined = pd.merge(
        fc_period_2023[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
        fc_period_2024[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
        on='ds',
        suffixes=('_2023', '_2024')
    )
    
    forecast_combined['yhat_avg'] = (forecast_combined['yhat_2023'] + forecast_combined['yhat_2024']) / 2
    forecast_combined['yhat_lower_avg'] = (forecast_combined['yhat_lower_2023'] + forecast_combined['yhat_lower_2024']) / 2
    forecast_combined['yhat_upper_avg'] = (forecast_combined['yhat_upper_2023'] + forecast_combined['yhat_upper_2024']) / 2
    
    plt.figure(figsize=(10, 6))
    plt.plot(forecast_combined['ds'], forecast_combined['yhat_avg'], label='Average Prediction', color='blue')
    plt.fill_between(forecast_combined['ds'],
                     forecast_combined['yhat_lower_avg'],
                     forecast_combined['yhat_upper_avg'],
                     color='blue', alpha=0.2, label='Prediction Interval')
    plt.title(f"Combined Forecast for {forecast_start} to {forecast_end} (2025)")
    plt.xlabel("Date")
    plt.ylabel("Predicted Value")
    plt.legend()
    plt.show()
    
    return forecast_combined

# --- Main Execution ---

def main():
    # Forecast for the target future period in 2025 based on parallel models trained on 2023 and 2024
    dirty_data = data_processing.load_data()
    clean_file = data_processing.clean_data(dirty_data)
    forecast_combined = predict_future_parallel(clean_file, forecast_start="2025-04-01", forecast_end="2025-04-8")
    
if __name__ == '__main__':
    main()