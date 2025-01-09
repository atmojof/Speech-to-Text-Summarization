import os
from dotenv import load_dotenv
import streamlit as st
import azure.cognitiveservices.speech as speechsdk
from openai import OpenAI
import tempfile
from pydub import AudioSegment

# Load environment variables from .env file
load_dotenv()
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SERVICE_REGION = os.getenv("AZURE_SERVICE_REGION")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI API
openai_api = OpenAI(api_key=OPENAI_API_KEY)

def save_audio_file(uploaded_file):
    """Save the uploaded audio file to a temporary location."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio = AudioSegment.from_file(uploaded_file)
    audio.export(temp_file.name, format="wav")
    return temp_file.name

def azure_speech_to_text(audio_path):
    """Transcribe audio using Azure Speech Services."""
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SERVICE_REGION)
    audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    results = []

    def recognized_handler(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            st.write(f"DEBUG: Recognized text: {evt.result.text}")  # Debugging line
            results.append((evt.result.offset_in_ticks / 10_000_000, evt.result.text))  # Convert ticks to seconds

    def session_stopped_handler(evt):
        st.write("DEBUG: Session stopped.")  # Debugging line

    def canceled_handler(evt):
        st.write(f"DEBUG: Canceled reason: {evt.reason}")  # Debugging line

    speech_recognizer.recognized.connect(recognized_handler)
    speech_recognizer.session_stopped.connect(session_stopped_handler)
    speech_recognizer.canceled.connect(canceled_handler)

    st.write("DEBUG: Starting recognition.")  # Debugging line
    speech_recognizer.start_continuous_recognition()

    # Wait for recognition to complete
    import time
    time.sleep(10)  # Wait for a fixed duration; adjust as needed

    speech_recognizer.stop_continuous_recognition()

    if not results:
        st.write("DEBUG: No results captured.")  # Debugging line

    return [(offset, text) for offset, text in results]

def azure_microphone_to_text():
    """Transcribe live audio from the microphone using Azure Speech Services."""
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SERVICE_REGION)
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    result = speech_recognizer.recognize_once_async().get()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return [(0, result.text)]  # Single result with no offset
    elif result.reason == speechsdk.ResultReason.NoMatch:
        return "No speech could be recognized."
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        return f"Speech Recognition canceled: {cancellation_details.reason}"

def summarize_text(text):
    """Summarize text using OpenAI GPT."""
    response = openai_api.Completion.create(
        model="text-davinci-003",
        prompt=f"Summarize the following text:\n{text}",
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Streamlit App
st.title("Speech-to-Text Transcription and Summarization")

# Input Method Selection
input_method = st.radio("Choose Input Method", ("Upload File", "Use Microphone"))

if input_method == "Upload File":
    uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "m4a"])
else:
    st.write("Using Microphone for real-time transcription.")

# Model Selection
model_choice = st.radio("Choose Transcription Model", ("Azure", "OpenAI"))

# Submit Button
if st.button("Submit"):
    transcription_result = ""
    summary_result = ""

    if input_method == "Upload File" and uploaded_file:
        audio_path = save_audio_file(uploaded_file)

        if model_choice == "Azure":
            transcription_result = azure_speech_to_text(audio_path)
        else:
            transcription_result = summarize_text(openai_api.transcribe(audio_path))

    elif input_method == "Use Microphone":
        if model_choice == "Azure":
            transcription_result = azure_microphone_to_text()

    # Display Results
    if transcription_result:
        st.write("### Transcription Result:")
        for timestamp, text in transcription_result:
            st.write(f"[{timestamp:.2f}s] {text}")

        st.write("### Summary Result:")
        summary_result = summarize_text(" ".join(text for _, text in transcription_result))
        st.write(summary_result)
    else:
        st.write("No transcription result available.")