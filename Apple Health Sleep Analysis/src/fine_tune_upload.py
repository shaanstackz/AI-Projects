from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

file = client.files.create(
    file=open("/Users/shaankohli/Documents/Sleep Analysis/Apple Health Sleep Analysis/notebooks/data/sleep_train.jsonl", "rb"),
    purpose="fine-tune"
)

print("✅ File uploaded successfully!")
print("File ID:", file.id)

fine_tune = client.fine_tuning.jobs.create(
    training_file=file.id,
    model="gpt-3.5-turbo-0125"
)

print("🎯 Fine-tuning started! Job ID:", fine_tune.id)

job = client.fine_tuning.jobs.retrieve(fine_tune.id)
print("Job status:", job.status)
