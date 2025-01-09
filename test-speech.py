import os
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk

speech_key, service_region = "601c0ddb54cd40709ed8efe586d8ed42", "southeastasia"

speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)