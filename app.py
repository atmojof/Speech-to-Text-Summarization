import streamlit as st
import azure.cognitiveservices.speech as speechsdk
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
import openai
import os

# Configuration for Azure API
AZURE_SPEECH_KEY = "YOUR_AZURE_SPEECH_KEY"
AZURE_SERVICE_REGION = "YOUR_SERVICE_REGION"
AZURE_OPENAI_KEY = "YOUR_AZURE_OPENAI_KEY"

# Function: Speech-to-Text with Azure
def azure_speech_to_text(audio_path):
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SERVICE_REGION)
    audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
    speech_recognizer = speechsdk.ConversationTranscriber(speech_config=speech_config, audio_config=audio_config)

    all_transcriptions = []

    def recognized_handler(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            speaker_id = evt.result.speaker_id or "Unknown"
            text = evt.result.text
            all_transcriptions.append(f"Speaker {speaker_id}: {text}")

    speech_recognizer.recognized.connect(recognized_handler)

    try:
        speech_recognizer.start_transcribing()
        speech_recognizer.stop_transcribing()
    except Exception as e:
        return f"Error during Azure transcription: {e}"

    return "\n".join(all_transcriptions)

# Function: Speech-to-Text with Whisper
def whisper_speech_to_text(audio_path):
    try:
        # Initialize Whisper model
        model = pipeline("automatic-speech-recognition", model="openai/whisper-base")
        transcription = model(audio_path)
        return transcription["text"]
    except Exception as e:
        return f"Error during Whisper transcription: {e}"

# Function: Text Summarization
def summarize_text(input_text):
    openai.api_key = AZURE_OPENAI_KEY
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Summarize the following meeting transcript:\n\n{input_text}",
            max_tokens=100,
            temperature=0.5
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Error during summarization: {e}"

# Streamlit UI
st.title("Speech-to-Text with Summarization")
st.write("Upload an audio file to transcribe and summarize its content. Choose your preferred model for transcription.")

# Choose the model
model_choice = st.radio("Choose Speech-to-Text Model:", ("Azure", "Whisper"))

# Upload audio file
uploaded_file = st.file_uploader("Upload Audio File (e.g., .m4a, .wav)", type=["m4a", "wav", "mp3"])

if uploaded_file:
    # Save uploaded file to a temporary path
    audio_path = os.path.join("temp_audio", uploaded_file.name)
    os.makedirs("temp_audio", exist_ok=True)
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Perform Speech-to-Text
    st.write(f"### Transcription in Progress using {model_choice}...")
    if model_choice == "Azure":
        text_result = azure_speech_to_text(audio_path)
    elif model_choice == "Whisper":
        text_result = whisper_speech_to_text(audio_path)

    # Display transcription results
    if text_result:
        st.write("### Transcription Result:")
        st.text_area("Transcribed Text", text_result, height=300)

        # Perform Summarization
        st.write("### Summarization in Progress...")
        summary_result = summarize_text(text_result)
        st.write("### Summary Result:")
        st.text_area("Summary Text", summary_result, height=150)

    # Cleanup temporary audio file
    os.remove(audio_path)
