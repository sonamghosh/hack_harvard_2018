import requests
import json
from watson_developer_cloud import ToneAnalyzerV3
from watson_developer_cloud import WatsonApiException


def emotion_analyzer(text):
	# Call Tone Analyzer API from IBM Watson
	tone_analyzer = ToneAnalyzerV3(
		version='2018-10-15',
		username = 'c82494a5-9134-4fa4-b59a-4856e81aadc5',
		password = 'qJ0DFFFam4be',
		url = 'https://gateway.watsonplatform.net/tone-analyzer/api')

	# Print Result
	tone_analysis = tone_analyzer.tone({'text': text}, 'application/json').get_result()
	print('Input text = ', text, '\n')
	#print(json.dumps(tone_analysis, indent=2))
	#json_file = json.dumps(tone_analysis, indent=2)
	tones = tone_analysis['document_tone']['tones']
	items = [x['score'] for x in tones]
	max_score = max(items)
	max_emotion = items.index(max_score)

	return tones[max_emotion]["tone_name"]

if __name__ == "__main__":
	text = "Without you, I feel broke\
		Like I\'m half of a whole\
		Without you, I\'ve got no hand to hold\
		Without you, I feel torn\
		Like a sail in a storm\
		Without you, I\'m just a sad song\
		I\'m just a sad song"

	em = emotion_analyzer(text)
	print(em)
