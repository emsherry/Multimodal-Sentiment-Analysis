# utils/transcribe.py

import whisper

# Initialize the whisper model
model = whisper.load_model("base")  # You can choose a different model if needed

def transcribe_audio(file_path):
    """
    This function uses Whisper to transcribe the audio to text.
    """
    result = model.transcribe(file_path)
    return result['text']
