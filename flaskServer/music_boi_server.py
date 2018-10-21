#Imports
from flask import Flask, request
from flask_cors import CORS
import json
#from text_emotion_analyzer.py import emotion_analyzer
#from music_boi.py import music_boi

#Set app for Flask
app = Flask(__name__)
CORS(app)

#Routes
@app.route('/')
def hello_world():
    return 'Hello, World!'

#Receive string from HTTP POST and return midi
@app.route('/submitString', methods=['POST'])
def submitString():
    text = request.form['text']
    #emotion = emotion_analyzer(text)
    #midi = music_boi(emotion)
    cool = {"text": "GOTEM! " + text}
    cool = json.dumps(cool)
    print("Received: " + text)
    return cool
