import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def engineer_features(daily_df, sleep_df):
    sleep_times = sleep_df.groupby('date')['startDate'].min().reset_index()
    wake_times = sleep_df.groupby('date')['endDate'].max().reset_index()

    df = daily_df.copy()
    df['bedtime_hour'] = sleep_times['startDate'].dt.hour.values
    df['wake_hour'] = wake_times['endDate'].dt.hour.values

    df['prev_sleep'] = df['total_sleep_hours'].shift(1)
    df['sleep_debt'] = 8 - df['total_sleep_hours']
    df['rolling_7d_avg'] = df['total_sleep_hours'].rolling(7).mean()
    df['rolling_7d_std'] = df['total_sleep_hours'].rolling(7).std()

    df['weekday'] = pd.to_datetime(df['date']).dt.dayofweek  

    df = df.drop(columns=['date'])

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna()

    return df


def train_sleep_model(feature_df, target_column='total_sleep_hours'):
    # Drop columns safely
    cols_to_drop = [col for col in ['date', target_column] if col in feature_df.columns]
    features = feature_df.drop(columns=cols_to_drop)
    target = feature_df[target_column]

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"Model trained! Test MAE: {mae:.2f} hours")

    return model, X_test, y_test, preds


def generate_recommendations(feature_df, model):
    latest = feature_df.iloc[-1].copy()

    cols_to_drop = [col for col in ['date', 'total_sleep_hours'] if col in latest.index]
    features = latest.drop(cols_to_drop).values.reshape(1, -1)

    pred_sleep = model.predict(features)[0]

    recs = []

    if pred_sleep < 7:
        recs.append(f"Predicted sleep next night is {pred_sleep:.1f}h. Try going to bed earlier or reducing screen time.")
    elif pred_sleep > 9:
        recs.append(f"Predicted sleep next night is {pred_sleep:.1f}h. Ensure you are not oversleeping; maintain routine.")
    else:
        recs.append(f"Predicted sleep next night is {pred_sleep:.1f}h. Your sleep looks healthy!")

    return recs, pred_sleep
