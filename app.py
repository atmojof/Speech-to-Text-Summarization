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



# Function to convert m4a to wav using temp file
def convert_m4a_to_wav(input_file):
    audio = AudioSegment.from_file(input_file)
    temp_wav_path = "./temp_audio.wav"
    audio.export(temp_wav_path, format="wav")
    return temp_wav_path

# Function to transcribe with segment timestamps (Azure)
def recognize_with_segment_timestamps(audio_file_path, language):
    #AZURE_SPEECH_KEY = "your_speech_service_key"
    #AZURE_SERVICE_REGION = "your_service_region"

    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SERVICE_REGION)
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

# Streamlit UI
def main():
    st.title("Audio Transcription")

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
    
    # Get the language code from the selected language
    language_code = list(language_dict.keys())[list(language_dict.values()).index(language_choice)]

    # Filter to choose file type
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "x-m4a", "mp4"])
    #st.write(audio_file.type)

    if audio_file:
        if audio_file.type in ["audio/m4a", "audio/x-m4a", "audio/mp4", "video/mp4"]:
            # Convert m4a to wav
            st.write("Converting audio file...")
            audio_file_path = convert_m4a_to_wav(audio_file)
        else:
            # Save audio file as .wav
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav_file:
                with open(temp_wav_file.name, "wb") as f:
                    f.write(audio_file.getbuffer())
                audio_file_path = temp_wav_file.name

        # Submit button to trigger the transcription
        if st.button("Submit"):
            st.write("Transcribing...")
            transcription_result = recognize_with_segment_timestamps(audio_file_path, language_code)
            st.subheader("Transcription Result")
            st.text_area("", transcription_result, height=300)
            
            # Clean up the temporary file
            os.remove(audio_file_path)

# Run Streamlit app
if __name__ == "__main__":
    main()