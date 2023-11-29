import whisper

model = whisper.load_model("base")
result = model.transcribe("static/audio/audio.mp3")
print(result["text"])
