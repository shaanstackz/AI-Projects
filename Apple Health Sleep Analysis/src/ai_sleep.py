import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def engineer_features(daily_sleep, sleep_df=None):
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
    df["avg_14d_sleep"] = df["sleep_duration"].rolling(window=14, min_periods=1).mean()
    df["sleep_debt"] = (8 - df["sleep_duration"]).clip(lower=0)
    df["sleep_consistency"] = df["sleep_duration"].rolling(window=7, min_periods=1).std().fillna(0)

    if sleep_df is not None and {"startDate", "endDate"}.issubset(sleep_df.columns):
        sleep_df["startDate"] = pd.to_datetime(sleep_df["startDate"])
        sleep_df["endDate"] = pd.to_datetime(sleep_df["endDate"])
        sleep_df["date"] = sleep_df["endDate"].dt.date
        sleep_summary = (
            sleep_df.groupby("date")
            .agg(bedtime_hour=("startDate", lambda x: x.dt.hour.mean()),
                 wake_hour=("endDate", lambda x: x.dt.hour.mean()))
            .reset_index()
        )
        df = df.merge(sleep_summary, how="left", left_on="date", right_on="date")
        df["bedtime_variability"] = df["bedtime_hour"].diff().abs().fillna(0)
    else:
        df["bedtime_hour"] = np.nan
        df["wake_hour"] = np.nan
        df["bedtime_variability"] = np.nan

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["weekday"] = df["date"].dt.weekday
        df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    else:
        df["is_weekend"] = 0

    if "time_in_bed" in df.columns:
        df["sleep_efficiency"] = df["sleep_duration"] / (df["time_in_bed"] + 1e-6)
    else:
        df["sleep_efficiency"] = np.where(df["sleep_duration"] > 0, 1.0, 0)

    df = df.fillna(df.mean(numeric_only=True))

    print(f"✅ Using '{found}' as sleep duration column with advanced features added.")
    print(f"📊 Generated columns: {', '.join(df.columns)}")
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
    latest = df.iloc[[-1]].copy()

    feature_cols = model.feature_names_in_
    for col in feature_cols:
        if col not in latest.columns:
            latest[col] = 0
    latest = latest[feature_cols]

    pred_sleep = model.predict(latest)[0]
    current_sleep = df["sleep_duration"].iloc[-1]
    diff = pred_sleep - current_sleep

    recs = []

    if diff > 0.25:
        recs.append(f"Predicted to sleep {pred_sleep:.1f}h tonight. You're currently getting {current_sleep:.1f}h, consider going to bed earlier.")
    elif diff < -0.25:
        recs.append(f"Predicted to sleep {pred_sleep:.1f}h tonight. You're currently getting {current_sleep:.1f}h, consider lighter evening activity or earlier wake-up.")
    else:
        recs.append(f"Your predicted sleep of {pred_sleep:.1f}h is consistent with current patterns. Keep it up!")

    consistency = df["sleep_consistency"].iloc[-1] if "sleep_consistency" in df.columns else 0
    if consistency > 1.5:
        recs.append(f"Your 7-day sleep consistency is {consistency:.1f}h. Try keeping a more consistent bedtime.")

    variability = df["bedtime_variability"].iloc[-1] if "bedtime_variability" in df.columns else 0
    if variability > 1:
        recs.append(f"Your bedtime varies by {variability:.1f}h. Aim for a steady bedtime for better rest quality.")

    debt = df["sleep_debt"].iloc[-1] if "sleep_debt" in df.columns else 0
    if debt > 1:
        recs.append(f"You have a sleep debt of {debt:.1f}h. Prioritize extra rest when possible.")

    if "is_weekend" in df.columns:
        if df["is_weekend"].iloc[-1] == 0 and current_sleep < 7:
            recs.append("It's a weekday and you're getting less than 7h of sleep. Consider adjusting your evening routine.")

    if "sleep_efficiency" in df.columns:
        efficiency = df["sleep_efficiency"].iloc[-1]
        if efficiency < 0.85:
            recs.append(f"Your sleep efficiency is {efficiency:.2f}. Minimize disturbances during sleep for better quality.")

    return recs, pred_sleep
