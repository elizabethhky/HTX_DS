import os
import requests
import pandas as pd
from tqdm import tqdm

# ==== Paths ====
CV_FOLDER = "common_voice"
CSV_PATH = os.path.join(CV_FOLDER, "cv-valid-dev.csv")
# Adjust if mp3 files are in a different path
AUDIO_FOLDER = os.path.join(CV_FOLDER, "cv-valid-dev")

# ==== Load CSV ====
df = pd.read_csv(CSV_PATH)

# Add column to store generated transcriptions
df["generated_text"] = ""

# Loop through each audio file and call ASR API
for idx, row in tqdm(df.iterrows(), total=len(df)):
    file_name = row["filename"]
    audio_path = os.path.join(AUDIO_FOLDER, file_name)

    if not os.path.exists(audio_path):
        print(f"File missing: {file_name}")
        continue

    with open(audio_path, "rb") as f:
        files = {"file": (file_name, f, "audio/mpeg")}
        try:
            response = requests.post("http://localhost:8001/asr", files=files)
            if response.status_code == 200:
                result = response.json()
                df.at[idx, "generated_text"] = result.get("transcription", "")
            else:
                print(f"API failed on {file_name}: {response.status_code}")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

# Save updated CSV
df.to_csv(CSV_PATH, index=False)
print("✅ Transcriptions saved to cv-valid-dev.csv")
