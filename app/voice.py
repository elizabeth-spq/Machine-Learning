import pyttsx3

text_speech = pyttsx3.init()

answer = input("What you want to convert in a voice: ")

text_speech.say(answer)
text_speech.runAndWait()