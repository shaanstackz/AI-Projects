import pandas as pd

def summarize_sleep(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize total sleep per night."""
    df['date'] = df['startDate'].dt.date
    daily = (
        df.groupby(['date', 'sleep_type'])['duration_hours']
        .sum().unstack(fill_value=0).reset_index()
    )
    daily['total_sleep_hours'] = daily.sum(axis=1, numeric_only=True)
    return daily

def calculate_consistency(daily_df: pd.DataFrame) -> dict:
    """Return sleep consistency statistics."""
    avg = daily_df['total_sleep_hours'].mean()
    std = daily_df['total_sleep_hours'].std()
    cv = std / avg * 100
    return {'average_hours': avg, 'std_dev': std, 'consistency_%': cv}
