import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def ai_analyze_sleep_patterns(df: pd.DataFrame, model=os.getenv("GPT_MODEL"), mode="coach"):
    if df.empty:
        return "No data available for analysis."
    sample_data = df.head(10).to_markdown(index=False)
    if mode == "coach":
        summary_prompt = f"""
        You are a friendly sleep coach. Analyze this summary of daily sleep data.

        Data sample:
        {sample_data}

        Provide 3-5 concise bullet points:
        - Overall sleep quality and patterns
        - How exercise, heart rate, or habits influence sleep
        - One practical tip for improvement
        Keep it short and encouraging.
        """
    elif mode == "scientist":
        summary_prompt = f"""
        You are a data scientist. Perform a detailed correlation analysis 
        of daily sleep, heart rate, and exercise data.

        Data sample:
        {sample_data}

        Provide:
        - Key statistical patterns
        - Correlated factors with good/bad sleep
        - Data anomalies or outliers
        - Improvement suggestions
        """
    elif mode == "story":
        summary_prompt = f"""
        You are a creative AI writing a short reflection called 'My Sleep Story This Week'.

        Use the data summary below to write 3-4 sentences describing the user’s sleep journey.
        Be poetic but grounded in the numbers.

        Data sample:
        {sample_data}
        """
    else:
        raise ValueError("Invalid mode. Choose 'coach', 'scientist', or 'story'.")
    response = client.responses.create(
        model=model,
        input=summary_prompt,
        temperature=0.7,
    )
    return response.output[0].content[0].text

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
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df

def train_sleep_model(feature_df, target_column='total_sleep_hours'):
    features = feature_df.drop(columns=[target_column]).copy()
    for col in features.select_dtypes(include=['datetime64[ns]']).columns:
        features[col] = features[col].view('int64') / 1e9
    target = feature_df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"✅ Model trained successfully! Test MAE: {mae:.2f} hours")
    return model, X_test, y_test, preds

def generate_recommendations(df, model):
    latest = df.iloc[[-1]].copy()
    feature_cols = model.feature_names_in_
    for col in feature_cols:
        if col not in latest.columns:
            latest[col] = 0
    for col in latest.select_dtypes(include=['datetime64[ns]']).columns:
        latest[col] = latest[col].view('int64') / 1e9
    latest = latest[feature_cols]
    pred_sleep = model.predict(latest)[0]
    current_sleep = df['total_sleep_hours'].iloc[-1]
    diff = pred_sleep - current_sleep
    recs = []
    if diff > 0.25:
        recs.append(f"Predicted sleep: {pred_sleep:.1f}h. You're currently at {current_sleep:.1f}h — try going to bed earlier.")
    elif diff < -0.25:
        recs.append(f"Predicted sleep: {pred_sleep:.1f}h. You may be oversleeping slightly — consider lighter evening activity.")
    else:
        recs.append(f"Your predicted sleep of {pred_sleep:.1f}h matches your current average.")
    if 'sleep_debt' in df.columns:
        debt = df['sleep_debt'].iloc[-1]
        if debt > 1:
            recs.append(f"You have a sleep debt of {debt:.1f}h. Try a weekend catch-up nap.")
    if 'bedtime_variability' in df.columns:
        var = df['bedtime_variability'].iloc[-1]
        if var > 1:
            recs.append(f"Your bedtime varies by {var:.1f}h. Consistency helps improve deep sleep.")
    if 'heart_rate' in df.columns and df['heart_rate'].iloc[-1] > 75:
        recs.append("Elevated heart rate before sleep may reduce quality — try relaxation or breathing exercises.")
    return recs, pred_sleep
