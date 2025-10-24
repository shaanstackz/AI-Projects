import json
import pandas as pd
import os

df = pd.read_csv("/Users/shaankohli/Documents/Sleep Analysis/Apple Health Sleep Analysis/notebooks/results/daily_sleep_summary.csv")

os.makedirs("fine_tuning", exist_ok=True)
output_path = "fine_tuning/sleep_finetune.jsonl"

with open(output_path, "w") as f:
    for _, row in df.iterrows():
        prompt = (
            f"Analyze this sleep data: total sleep {row.get('TotalSleep', 'N/A')}h, "
            f"deep sleep {row.get('DeepSleep', 'N/A')}h, rem {row.get('REM', 'N/A')}h, "
            f"heart rate {row.get('AvgHeartRate', 'N/A')} bpm, steps {row.get('Steps', 'N/A')}."
        )

        completion = (
            "You had good recovery overall. Try keeping consistent sleep times "
            "and aim to improve your deep sleep slightly."
        )

        record = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion}
            ]
        }

        f.write(json.dumps(record) + "\n")

print(f"✅ JSONL training file created: {output_path}")
