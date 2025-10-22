import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def engineer_features(daily_sleep, sleep_df):
    df = daily_sleep.copy()

    possible_names = ["sleep_duration", "total_sleep_hours", "asleep_duration", "sleep_hours", "duration"]
    found = None
    for name in possible_names:
        if name in df.columns:
            found = name
            break

    if found is None:
        raise ValueError(f"No sleep duration column found. Available columns: {df.columns.tolist()}")

    df.rename(columns={found: "sleep_duration"}, inplace=True)

    df["sleep_change"] = df["sleep_duration"].diff().fillna(0)
    df["avg_7d_sleep"] = df["sleep_duration"].rolling(window=7, min_periods=1).mean()

    if "time_in_bed" in df.columns:
        df["sleep_efficiency"] = df["sleep_duration"] / (df["time_in_bed"] + 1e-6)
    else:
        df["sleep_efficiency"] = df["sleep_duration"] / (df["sleep_duration"] + 1)  # neutral baseline

    if "date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["weekday"] = df["date"].dt.weekday
        df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    else:
        df["is_weekend"] = 0

    if sleep_df is not None and "heart_rate_mean" in sleep_df.columns:
        df = df.merge(sleep_df[["date", "heart_rate_mean"]], on="date", how="left")

    df = df.fillna(df.mean(numeric_only=True))

    print(f"✅ Using '{found}' as sleep duration column.")
    return df



def train_sleep_model(df):
    features = ["sleep_duration", "sleep_change", "avg_7d_sleep", 
                "sleep_efficiency", "is_weekend", "heart_rate_mean"]
    features = [f for f in features if f in df.columns]

    X = df[features]
    y = df["sleep_duration"].shift(-1).fillna(df["sleep_duration"].mean())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"Model R²: {r2_score(y_test, preds):.3f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.2f} hours")

    return model, X_test, y_test, preds


def generate_recommendations(df, model):
    latest = df.iloc[-1:].copy()
    if "sleep_change" not in latest.columns:
        latest["sleep_change"] = 0

    pred_sleep = model.predict(latest[[col for col in latest.columns if col in model.feature_names_in_]])[0]
    current_sleep = latest["sleep_duration"].values[0]

    diff = pred_sleep - current_sleep

    recs = []
    if diff > 0.5:
        recs.append("Your predicted sleep duration is expected to increase. Keep up your routine!")
    elif diff < -0.5:
        recs.append("Predicted sleep duration might drop. Try reducing screen time before bed.")
    else:
        recs.append("Your sleep duration looks consistent — maintain your current habits.")

    if "sleep_efficiency" in latest.columns and latest["sleep_efficiency"].values[0] < 0.85:
        recs.append("Low sleep efficiency detected. Avoid caffeine after 4 PM and optimize your environment.")

    if "heart_rate_mean" in latest.columns and latest["heart_rate_mean"].values[0] > 70:
        recs.append("Elevated heart rate during sleep. Consider stress-reducing techniques before bed.")

    return recs, pred_sleep
