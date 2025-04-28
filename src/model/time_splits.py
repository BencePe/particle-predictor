from datetime import datetime, date, timedelta
from pyspark.sql import functions as F

def hybrid_time_validation(df, datetime_col="datetime", 
                          initial_train_years=1, 
                          test_window_months=3,
                          gap_weeks=2):
    """
    Creates sequential splits with:
    - Initial 1-year training period
    - 3-month test windows
    - 2-week gap between train/test
    """


    splits = []

    min_dt = df.select(F.min(datetime_col)).first()[0]
    if isinstance(min_dt, date) and not isinstance(min_dt, datetime):
        min_dt = datetime.combine(min_dt, datetime.min.time())

    current_train_end = min_dt + timedelta(days=initial_train_years * 365)

    while True:
        test_start = current_train_end + timedelta(weeks=gap_weeks)
        test_end = test_start + timedelta(days=test_window_months * 30)

        max_dt = df.select(F.max(datetime_col)).first()[0]
        if isinstance(max_dt, date) and not isinstance(max_dt, datetime):
            max_dt = datetime.combine(max_dt, datetime.min.time())

        if test_end > max_dt:
            break

        train = df.filter(F.col(datetime_col) <= current_train_end)
        test = df.filter(
            (F.col(datetime_col) >= test_start) &
            (F.col(datetime_col) < test_end)
        )

        splits.append((train, test))
        current_train_end = test_end

    return splits
