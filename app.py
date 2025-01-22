import os
import requests
import json
from dotenv import load_dotenv
import streamlit as st
import azure.cognitiveservices.speech as speechsdk
import openai
from openai import OpenAIError 
import tempfile
from pydub import AudioSegment
from transformers import pipeline
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient

import google.generativeai as genai

load_dotenv()

AZURE_KEY = os.getenv("AZURE_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_SERVICE_REGION = os.getenv("AZURE_SERVICE_REGION")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure Google API for audio summarization
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Azure Blob Storage configuration
AZURE_STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=smlbird;AccountKey=yE8Ps/0tzr9JxaHVEsG70SG8RwkBap3PJ+7Y4wOcyLb3Qtg7+1DLd/xY+IRxNnilJUTjfLq2BYRM+AStdEPHjw==;EndpointSuffix=core.windows.net"
CONTAINER_NAME = "input"

# Initialize OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")


# Functions =================================================================>
# Function to convert m4a to wav using temp file =================================================================
def convert_m4a_to_wav(input_file):
    audio = AudioSegment.from_file(input_file)
    temp_wav_path = "./temp_audio.wav"
    audio.export(temp_wav_path, format="wav")
    return temp_wav_path

# Function to upload file to Azure Blob Storage and get the URL =================================================================
def upload_to_blob_storage(file_path, file_name):
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=file_name)
    
    with open(file_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    
    blob_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{CONTAINER_NAME}/{file_name}"
    return blob_url

# Google Transcription
def gemini_audio_to_summary(audio_file_path):
    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
    audio_file = genai.upload_file(path=audio_file_path)
    response = model.generate_content(
        [
            "Please transcribe the following audio.",
            audio_file
        ]
    )
    return response.text

# Function to transcribe with segment timestamps (Azure) =================================================================
def azure_audio_to_text(audio_file_path, language):
    #AZURE_KEY = "your_speech_service_key"
    #AZURE_SERVICE_REGION = "your_service_region"

    speech_config = speechsdk.SpeechConfig(subscription=AZURE_KEY, region=AZURE_SERVICE_REGION)
    speech_config.speech_recognition_language = language

    # Request detailed output
    speech_config.output_format = speechsdk.OutputFormat.Detailed

    audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    transcriptions = []

    print("Processing the audio file for transcription with timestamps...")

    def recognized_callback(evt):
        # Access the detailed recognition result
        result_json = evt.result.json
        result_dict = eval(result_json)  # Convert JSON string to a dictionary

        recognized_text = result_dict["DisplayText"]
        offset = result_dict["Offset"]  # Start time in 100-nanoseconds
        duration = result_dict["Duration"]  # Duration in 100-nanoseconds

        # Convert offset to MM:SS format
        start_seconds = offset / 10**7
        start_minutes = int(start_seconds // 60)
        start_seconds = int(start_seconds % 60)

        # Store transcription with timestamp
        transcriptions.append(f"[{start_minutes:02}:{start_seconds:02}] {recognized_text}")

        # Update the transcription text area in Streamlit
        st.session_state.transcription_text += f"[{start_minutes:02}:{start_seconds:02}] {recognized_text}\n"
        st.text_area("Transcription", st.session_state.transcription_text, height=300)

    done = False

    def stop_cb(evt):
        nonlocal done
        done = True

    # Connect callbacks
    speech_recognizer.recognized.connect(recognized_callback)
    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)

    # Start continuous recognition
    speech_recognizer.start_continuous_recognition()
    while not done:
        pass
    speech_recognizer.stop_continuous_recognition()

    return "\n".join(transcriptions)

# Fast transcription
def fast_transcribe(audio_file_url, language):
    url = f"https://{AZURE_SERVICE_REGION}.api.cognitive.microsoft.com/speechtotext/v3.2/transcriptions"
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "contentUrls": [audio_file_url],
        "locale": language,
        "displayForm": True,
        "properties": {
            "wordLevelTimestampsEnabled": True,
            "punctuationMode": "DictatedAndAutomatic",
            "profanityFilterMode": "Masked"
        }
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        return response.json()  # Return the JSON response
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        print(f"Response content: {response.content.decode('utf-8')}")
        return {"error": f"HTTP error occurred: {http_err}"}  # Return error as JSON
    except Exception as err:
        print(f"An error occurred: {err}")
        return {"error": f"An error occurred: {err}"}  # Return error as JSON


#================================================================>
def summarize_transcription(document, context=""):
    text_analytics_client = TextAnalyticsClient(endpoint=AZURE_ENDPOINT, credential=AzureKeyCredential(AZURE_KEY))
    
    # Combine document with context if provided
    full_text = document
    if context:
        full_text = context + "\n\n" + document
    
    # Abstractive Summarization
    #poller = text_analytics_client.begin_abstract_summary([document])
    poller = text_analytics_client.begin_abstract_summary([{"id": "1", "text": full_text}])
    abstract_summary_results = poller.result()
    
    abstractive_summaries = []
    for result in abstract_summary_results:
        if result.kind == "AbstractiveSummarization":
            abstractive_summaries = " ".join([summary.text for summary in result.summaries])
        elif result.is_error:
            abstractive_summaries = "Error: {} - {}".format(result.error.code, result.error.message)
    
    # Extractive Summarization
    #poller = text_analytics_client.begin_extract_summary([document])
    poller = text_analytics_client.begin_extract_summary([{"id": "1", "text": full_text}])
    extract_summary_results = poller.result()
    
    extractive_summary = ""
    for result in extract_summary_results:
        if result.kind == "ExtractiveSummarization":
            extractive_summary = " ".join([sentence.text for sentence in result.sentences])
        elif result.is_error:
            extractive_summary = "Error: {} - {}".format(result.error.code, result.error.message)
    
    #return { "abstractive_summaries": abstractive_summaries, "extractive_summary": extractive_summary}
    return f"Abstractive Summaries:\n{abstractive_summaries}\n\nExtractive Summary:\n{extractive_summary}"

# Streamlit UI =================================================================>
def main():
    st.title("🔉Audio Transcription")

    # Initialize session state for transcription text
    if "transcription_text" not in st.session_state:
        st.session_state.transcription_text = ""

    # Language selection (human-readable names with language codes)
    language_dict = {
        "id-ID": "Indonesian (Indonesia)",
        "en-US": "English (US)",
        "es-ES": "Spanish (Spain)",
        "fr-FR": "French (France)"
    }
    language_choice = st.selectbox("Choose Language", list(language_dict.values()))

    # Model selection
    model_choice = st.selectbox("Choose Model", ("Azure AI", "Google Gemini"))
    
    # Get the language code from the selected language
    language_code = list(language_dict.keys())[list(language_dict.values()).index(language_choice)]

    # Option to summarize the transcription
    st.session_state.summarize_option = st.radio("Do you want to summarize the transcription?", ("No", "Yes"))

    if st.session_state.summarize_option == "Yes":
        context = st.text_area("Provide additional context (optional)")
    else:
        context = ""

    # Filter to choose file type
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "x-m4a", "mp4"])
    #st.write(audio_file.type)

    if audio_file: st.audio(audio_file, format=audio_file.type)

    # Submit
    submitted = st.button("Submit")

    if audio_file:
        
        
        if audio_file.type in ["audio/m4a", "audio/x-m4a", "audio/mp4", "video/mp4"]:
            # Convert m4a to wav
            with st.spinner("Converting audio file..."):
                audio_file_path = convert_m4a_to_wav(audio_file)
        else:
            # Save audio file as .wav
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav_file:
                with open(temp_wav_file.name, "wb") as f:
                    f.write(audio_file.getbuffer())
                audio_file_path = temp_wav_file.name

        # Submit button to trigger the transcription
        if submitted:
            # Upload the audio file to Azure Blob Storage
            #with st.spinner("Uploading audio file..."):
            #    audio_file_url = upload_to_blob_storage(audio_file_path, "temp_audio.wav")
            #st.write('ok')

            with st.spinner("Transcribing..."):

                if model_choice == "Azure AI":
                    transcription_result = azure_audio_to_text(audio_file_path, language_code)
                elif model_choice == "Google Gemini":
                    transcription_result = gemini_audio_to_summary(audio_file_path)

                #transcription_result = fast_transcribe(audio_file_url, language_code)
                st.subheader("Transcription Result")
                st.text_area("", transcription_result, height=300)
            
            # Check if there was an error in the transcription process
            if transcription_result:
                # Display transcription result
                st.subheader("Extractive Summary & Abstractive Summary")

                if st.session_state.summarize_option == "Yes":
                    with st.spinner('Summarizing...'):
                        summary = summarize_transcription(transcription_result, context)
                    st.text_area("Summary", summary, height=300)

# Run Streamlit app
if __name__ == "__main__":
    main()