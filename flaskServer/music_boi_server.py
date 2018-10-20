#Imports
from flask import Flask, request
from text_emotion_analyzer.py import emotion_analyzer
from music_boi.py import music_boi

#Set app for Flask
app = Flask(__name__)

#Routes
@app.route('/')
def hello_world():
    return 'Hello, World!'

#Receive string from HTTP POST and return midi
@app.route('/getString', methods=['POST'])
def getString():
    text = request.form['text']
    emotion = emotion_analyzer(text)
    midi = music_boi(emotion)
    return text
