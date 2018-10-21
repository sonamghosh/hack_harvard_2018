#Imports
from flask import Flask, request, send_file
from flask_cors import CORS
import dropbox
import json
import config

from text_emotion_analyzer import emotion_analyzer
#from music_boi.py import music_boi

#Set app for Flask
app = Flask(__name__)
CORS(app)

#Setup dropbox
dbx = dropbox.Dropbox(config.access_token)

#Routes
@app.route('/')
def hello_world():
    return "Hello World!"

#Receive string from HTTP POST and return midi
@app.route('/submitString', methods=['POST'])
def submitString():
    text = request.form['text']
    emotion = emotion_analyzer(text)
    #midi = music_boi(emotion)
    print("Received: " + text)
    filename = '/Steins;Gate-Believe-Me.mid'
    f = open('./Steins;Gate-Believe-Me.mid', 'rb')
    dbx.files_upload(bytes(f.read()), filename)
    link = dbx.sharing_create_shared_link(path=filename, short_url=True)
    cool = {"text": emotion, "link": link.url}
    cool = json.dumps(cool)
    return cool
