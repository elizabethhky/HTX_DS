from fastapi import FastAPI, UploadFile, File
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pydub import AudioSegment
import torch
import io
import numpy as np
import os
import uuid  # For generating unique file names

app = FastAPI()

# Load Model and Processor
MODEL_NAME = "facebook/wav2vec2-large-960h"
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)


@app.get("/ping")
def ping():
    return {"message": "pong"}


@app.post("/asr")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Save file to disk temporarily
        temp_filename = f"temp_{uuid.uuid4().hex}.mp3"
        with open(temp_filename, "wb") as temp_file:
            temp_file.write(await file.read())

        # Load audio using pydub
        audio = AudioSegment.from_file(temp_filename, format="mp3")
        audio = audio.set_frame_rate(16000).set_channels(1)

        samples = np.array(audio.get_array_of_samples()
                           ).astype(np.float32) / 32768.0
        input_tensor = torch.tensor(samples)

        input_values = processor(
            input_tensor, sampling_rate=16000, return_tensors="pt").input_values

        with torch.no_grad():
            logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        duration = round(len(audio) / 1000.0, 2)

        return {"transcription": transcription, "duration": duration}

    except Exception as e:
        return {"error": str(e)}

    finally:
        # Ensure file is deleted even if error happens
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
