import requests
import json
from watson_developer_cloud import ToneAnalyzerV3
from watson_developer_cloud import WatsonApiException


# Call Tone Analyzer API
tone_analyzer = ToneAnalyzerV3(
	version='2018-10-19',
	username = 'c82494a5-9134-4fa4-b59a-4856e81aadc5',
	password = 'qJ0DFFFam4be',
	url = 'https://gateway.watsonplatform.net/tone-analyzer/api')


# Dummy text
#text = input('Insert text here: ')
text = "Dan is a camel"
# Grab the result
tone_analysis = tone_analyzer.tone({'text': text},
	                                'application/json').get_result()

# Print Result
print('Input text = ', text, '\n')
print(json.dumps(tone_analysis, indent=2))	                               