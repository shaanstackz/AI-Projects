import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def engineer_features(daily_df, sleep_times, health_df=None):
    df = daily_df.copy()
    
    sleep_times = sleep_times.copy()
    sleep_times['startDate'] = pd.to_datetime(sleep_times['startDate'])
    sleep_times['endDate'] = pd.to_datetime(sleep_times['endDate'])
    sleep_times['date'] = sleep_times['startDate'].dt.normalize()
    
    bedtime_per_day = sleep_times.groupby('date')['startDate'].apply(lambda x: x.dt.hour.mean())
    wake_per_day = sleep_times.groupby('date')['endDate'].apply(lambda x: x.dt.hour.mean())
    
    if bedtime_per_day.index.tz is not None:
        bedtime_per_day.index = bedtime_per_day.index.tz_localize(None)
    if wake_per_day.index.tz is not None:
        wake_per_day.index = wake_per_day.index.tz_localize(None)
    
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    df = df.merge(bedtime_per_day.rename('bedtime_hour'), on='date', how='left')
    df = df.merge(wake_per_day.rename('wake_hour'), on='date', how='left')
    
    df['prev_sleep'] = df['total_sleep_hours'].shift(1)
    df['sleep_debt'] = 8 - df['total_sleep_hours']
    df['rolling_7d_avg'] = df['total_sleep_hours'].rolling(7).mean()
    df['weekday'] = df['date'].dt.day_name()
    df['weekday_num'] = df['date'].dt.weekday
    
    if health_df is not None:
        health_df = health_df.copy()
        health_df['startDate'] = pd.to_datetime(health_df['startDate'])
        health_df['date'] = health_df['startDate'].dt.normalize()
        agg = health_df.groupby(['date', 'type'])['value'].mean().unstack(fill_value=0)
        if agg.index.tz is not None:
            agg.index = agg.index.tz_localize(None)
        df = df.merge(agg, left_on='date', right_index=True, how='left')
    
    df.fillna(0, inplace=True)

    # Convert categorical columns to numeric (one-hot)
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df

def train_sleep_model(feature_df, target_column='total_sleep_hours'):
    features = feature_df.drop(columns=[target_column]).copy()
    
    for col in features.select_dtypes(include=['datetime64[ns]']).columns:
        features[col] = features[col].view('int64') / 1e9  # convert to seconds
    
    target = feature_df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"Model trained! Test MAE: {mae:.2f} hours")
    
    return model, X_test, y_test, preds

def generate_recommendations(df, model):
    latest = df.iloc[[-1]].copy()
    
    # Ensure all features exist
    feature_cols = model.feature_names_in_
    for col in feature_cols:
        if col not in latest.columns:
            latest[col] = 0
    
    # Convert datetime columns to numeric (seconds since epoch)
    for col in latest.select_dtypes(include=['datetime64[ns]']).columns:
        latest[col] = latest[col].view('int64') / 1e9
    
    latest = latest[feature_cols]
    
    pred_sleep = model.predict(latest)[0]
    current_sleep = df['total_sleep_hours'].iloc[-1]
    diff = pred_sleep - current_sleep
    
    recs = []
    
    if diff > 0.25:
        recs.append(f"Predicted to sleep {pred_sleep:.1f}h tonight. You're currently getting {current_sleep:.1f}h, consider going to bed earlier.")
    elif diff < -0.25:
        recs.append(f"Predicted to sleep {pred_sleep:.1f}h tonight. You're currently getting {current_sleep:.1f}h, consider lighter evening activity.")
    else:
        recs.append(f"Your predicted sleep of {pred_sleep:.1f}h is consistent with current patterns.")
    
    if 'sleep_consistency' in df.columns:
        consistency = df['sleep_consistency'].iloc[-1]
        if consistency > 1.5:
            recs.append(f"Your 7-day sleep consistency is {consistency:.1f}h. Aim for a more consistent bedtime.")
    
    if 'bedtime_variability' in df.columns:
        variability = df['bedtime_variability'].iloc[-1]
        if variability > 1:
            recs.append(f"Your bedtime varies by {variability:.1f}h. Try a steadier bedtime.")
    
    if 'sleep_debt' in df.columns:
        debt = df['sleep_debt'].iloc[-1]
        if debt > 1:
            recs.append(f"You have a sleep debt of {debt:.1f}h. Prioritize extra rest.")
    
    if 'activity' in df.columns and df['activity'].iloc[-1] > 5000:
        recs.append("High activity before bedtime may impact sleep. Consider lighter evening activity.")
    
    if 'heart_rate' in df.columns and df['heart_rate'].iloc[-1] > 75:
        recs.append("Elevated heart rate before sleep may reduce sleep quality. Try relaxation techniques.")
    
    if 'mindfulness' in df.columns and df['mindfulness'].iloc[-1] < 10:
        recs.append("Consider adding 10+ minutes of mindfulness or meditation before bedtime.")

    return recs, pred_sleep
