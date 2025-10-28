import pandas as pd
import json
import os

sleep_file = "/Users/shaankohli/Documents/Sleep Analysis/Apple Health Sleep Analysis/notebooks/results/daily_sleep_summary.csv"
output_file = "/Users/shaankohli/Documents/Sleep Analysis/Apple Health Sleep Analysis/notebooks/data/sleep_train.jsonl"

df = pd.read_csv(sleep_file)

df = df.fillna("N/A")

records = []

for _, row in df.iterrows():
    total_sleep = row.get("total_sleep_hours", "N/A")
    deep_sleep = row.get("deep_sleep_hours", "N/A")
    rem_sleep = row.get("rem_sleep_hours", "N/A")
    heart_rate = row.get("avg_heart_rate", "N/A")
    steps = row.get("total_steps", "N/A")

    user_prompt = (
        f"Analyze this sleep data: total sleep {total_sleep}h, "
        f"deep sleep {deep_sleep}h, REM {rem_sleep}h, "
        f"average heart rate {heart_rate} bpm, steps {steps}."
    )

    if total_sleep != "N/A" and float(total_sleep) < 6:
        ai_response = "You seem sleep-deprived. Try to increase your total sleep by at least 1 hour tonight."
    elif total_sleep != "N/A" and float(total_sleep) > 8:
        ai_response = "You’re getting excellent rest. Keep a consistent bedtime and stay active during the day."
    else:
        ai_response = "Your sleep looks balanced. Try maintaining this routine and monitor heart rate trends."

    records.append({
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": ai_response}
        ]
    })

os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")

print(f"✅ Generated {len(records)} training examples at {output_file}")
