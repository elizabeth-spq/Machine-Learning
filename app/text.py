import pytube
import whisper

youtubeVideoId = "https://www.youtube.com/watch?v=FLkt0mF0N38"
model=whisper.load_model('small')
youtubeVideo = pytube.YouTube(youtubeVideoId)
audio = youtubeVideo.streams.get_audio_only()
audio.download(filename='tmp.mp4')
result = model.transcribe('tmp.mp4')

print(result["text"])