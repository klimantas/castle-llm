import os
from dotenv import find_dotenv, load_dotenv
from datetime import datetime

# Get timestamp for filenames
timestamp = datetime.now().strftime("%Y%m%d")

print("----- DARTMOUTH MODELS -----")
load_dotenv(dotenv_path="dartmouth_key.env")
from langchain_dartmouth.llms import ChatDartmouth
chat = ChatDartmouth()
models = ChatDartmouth.list()

# Print and save Dartmouth models
dartmouth_models = []
for model in models:
    print(model.id)
    dartmouth_models.append(model.id)

# Save to file
with open(f"dartmouth_models.txt", "w") as f:
    f.write("Dartmouth Chat Models\n")
    f.write(f"Retrieved: {datetime.now().strftime('%Y-%m-%d')}\n")
    f.write("="*50 + "\n\n")
    for model_id in dartmouth_models:
        f.write(f"{model_id}\n")

print(f"\n✓ Saved {len(dartmouth_models)} Dartmouth models to dartmouth_models.txt")

print("\n----- OPEN AI MODELS -----")
load_dotenv(dotenv_path="openai_key.env")
from openai import OpenAI
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
models = client.models.list()

# Print and save OpenAI models
openai_models = []
for model in models.data:
    print(model.id)
    openai_models.append(model.id)

# Save to file
with open(f"openai_models.txt", "w") as f:
    f.write("OpenAI Models\n")
    f.write(f"Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*50 + "\n\n")
    for model_id in sorted(openai_models):  # Sort for easier reading
        f.write(f"{model_id}\n")

print(f"\n✓ Saved {len(openai_models)} OpenAI models to openai_models.txt")
