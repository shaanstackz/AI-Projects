from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

job_id = os.getenv("JOB_ID")

job = client.fine_tuning.jobs.retrieve(job_id)
print(f"Status: {job.status}")
print(f"Model: {job.fine_tuned_model}")
