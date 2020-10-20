from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import json
import os


authenticator = IAMAuthenticator('uDugZ3a7CQPGXBKzYSxf7te1mOq72NYWQroJ2lpsl_Ma')
speech_to_text = SpeechToTextV1(authenticator=authenticator )

speech_to_text.set_service_url('https://api.us-south.speech-to-text.watson.cloud.ibm.com/instances/124d399a-965d-4c3e-a073-953e74bfb471')



"""
with open('/Users/johnathontoh/Desktop/Task 9.1P/Resources_9.1/SpeechtoTextData/arctic_a0005.wav', 'rb') as audio_file:
    speech_recognition_results = speech_to_text.recognize(audio = audio_file, content_type='audio/wav').get_result()

print(json.dumps(speech_recognition_results, indent = 2))

# save file in JSON format
with open('/Users/johnathontoh/Desktop/Task 9.1P/Resources_9.1/SpeechtoTextDataJSON/arctic_a0005.json', 'w') as outfile:
    json.dump(speech_recognition_results, outfile)

with open('/Users/johnathontoh/Desktop/Task 9.1P/Resources_9.1/SpeechtoTextDataJSON/arctic_a0005.json') as infile:
    data = json.load(infile) # load data from a json file
print(data)
"""

AUDIO_DIR = '/Users/johnathontoh/Desktop/Task 9.1P/Resources_9.1/SpeechtoTextData/'
AUDIO_JSON_DIR = '/Users/johnathontoh/Desktop/Task 9.1P/Resources_9.1/SpeechtoTextDataJSON/'

"""
for files in os.listdir(AUDIO_DIR):
    if files.startswith('.') is False and files.endswith('.wav'):

        with open(AUDIO_DIR + files,'rb') as audio_file:
            speech_recognition_results_wav = speech_to_text.recognize(audio=audio_file, content_type='audio/wav').get_result()

        file_string_wav = files.split('.')

        with open(AUDIO_JSON_DIR + file_string_wav[0] +'.json','w') as outfile:
            json.dump(speech_recognition_results_wav, outfile)

    if files.startswith('.') is False and files.endswith('.flac'):

        with open(AUDIO_DIR + files,'rb') as audio_file:
            speech_recognition_results_flac = speech_to_text.recognize(audio=audio_file, content_type='audio/flac').get_result()

        file_string_flac = files.split('.')

        with open(AUDIO_JSON_DIR + file_string_flac[0] +'.json','w') as outfile:
            json.dump(speech_recognition_results_flac, outfile)
"""


for files in os.listdir(AUDIO_JSON_DIR):
    if files.startswith('.') is False:
        with open(AUDIO_JSON_DIR + files) as infile:
            data = json.load(infile)  # load data from a json file
        print(data, '\n')


