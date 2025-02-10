import streamlit as st
import time
import whisper
import tempfile
import os
from pydub import AudioSegment
import google.generativeai as genai

# Load the Whisper model globally
model = whisper.load_model("small")

# Configure Google API for audio summarization
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

@st.cache_data
def summarize_text(text):
    try:
        # Create a model instance
        model = genai.GenerativeModel('models/gemini-1.5-flash')
        
        # Define the prompt
        prompt = "Summarize the following text."
        
        # Generate content using the model
        response = model.generate_content(contents=f"{prompt}/n/n{text}")
        
        # Extract the summary from the response
        summary = response.text
        return summary
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Function to split large audio files
@st.cache_data
def split_audio(file_path, max_size_mb=25):
    audio = AudioSegment.from_file(file_path)
    max_size_bytes = max_size_mb * 1024 * 1024  # Convert MB to bytes
    file_size = len(audio.raw_data)

    if file_size <= max_size_bytes:
        return [file_path]

    # Split the file into chunks
    chunk_duration_ms = (max_size_bytes / file_size) * len(audio)  # Duration in ms
    chunks = [audio[i:i + int(chunk_duration_ms)] for i in range(0, len(audio), int(chunk_duration_ms))]
 
    # Save each chunk as a temporary file
    chunk_files = []
    for i, chunk in enumerate(chunks):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            chunk.export(temp_file.name, format="wav")
            chunk_files.append(temp_file.name)

    return chunk_files

# Function to transcribe audio
def transcribe_whisper1(file):
    # Save BytesIO to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name

    # Transcribe the audio file with timestamps
    result = model.transcribe(temp_file_path, word_timestamps=False)

    # Generate the transcription with timestamps
    transcription = "Transcription with Timestamps:\n"
    for segment in result['segments']:
        start_time = segment['start']
        text = segment['text']
        transcription += f"[{int(start_time // 60):02}:{int(start_time % 60):02}] {text}\n"
    return transcription

# Function to transcribe multiple audio files
def transcribe_whisper(files):
    transcription_results = ""

    for audio_file in files:
        # Save BytesIO to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_file.read())
            temp_file_path = temp_file.name

        # Split if the file is larger than 25MB
        audio_chunks = split_audio(temp_file_path, max_size_mb=25)

        # Transcribe each chunk
        transcription = ""
        for chunk_path in audio_chunks:
            result = model.transcribe(chunk_path, word_timestamps=False)
            for segment in result['segments']:
                start_time = segment['start']
                text = segment['text']
                transcription += f"[{int(start_time // 60):02}:{int(start_time % 60):02}] {text}\n"

        # Append the file name and transcription
        transcription_results += f"{audio_file.name}\n{transcription.strip()}\n\n"

    return transcription_results.strip()

# Function to display the login page
def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "researchsml" and password == "bsdserpong":
            st.session_state.logged_in = True
            st.success("Logged in successfully")
        else:
            st.error("Invalid username or password")
def app():
    st.title("🔉 Audio Transcription")

    # Initialize session state for transcription
    if "transcription_result" not in st.session_state:
        st.session_state.transcription_result = None
    if "summary_result" not in st.session_state:
        st.session_state.summary_result = None

    # Input-1: Model selection (currently only Whisper is implemented)
    model_choice = st.selectbox("Choose Model", ("Whisper (Relatively Faster)", "Azure AI (For More Accurate)"))

    # Language selection (human-readable names with language codes)
    language_dict = {
        "id-ID": "Indonesian (Indonesia)",
        "en-US": "English (US)",
        "es-ES": "Spanish (Spain)",
        "fr-FR": "French (France)"
    }
    language_choice = st.selectbox("Choose Language", list(language_dict.values()))
    language_code = list(language_dict.keys())[list(language_dict.values()).index(language_choice)]

    # Input-2: File uploader
    SUPPORTED_FORMATS = ["wav", "mp3", "ogg", "flac", "aac", "webm", "m4a"]
    audio_files = st.file_uploader(
        "Upload audio file(s) (Max 25MB each)", 
        type=SUPPORTED_FORMATS, 
        accept_multiple_files=True
    )

    # Submit
    if audio_files: 
        submitted = st.button("Submit")

        if submitted:

            start_time = time.time()
            with st.spinner("Transcribing..."):
                st.session_state.transcription_result = transcribe_whisper(audio_files)
            end_time = time.time()
            elapsed_time = end_time - start_time
            st.write(f"Transcription completed in {elapsed_time:.2f} seconds.")
                
            if st.session_state.transcription_result:
                # Display the combined transcription
                st.subheader("Transcription")
                st.text_area("", st.session_state.transcription_result, height=400)

                # Provide a download button
                st.download_button(
                    label="Download Transcription",
                    data=st.session_state.transcription_result,
                    file_name="transcription.txt",
                    mime="text/plain"
                )

                # Summarize the combined transcription
                with st.spinner("Summarizing transcriptions..."):
                    st.session_state.summary_result = summarize_text(st.session_state.transcription_result)

                # Display the transcription if it exists
                if st.session_state.summary_result:
                    st.text_area("Summary", st.session_state.summary_result, height=300)

                # Download button for summary
                st.download_button(
                    label="Download Summary",
                    data=st.session_state.summary_result,
                    file_name="summary.txt",
                    mime="text/plain"
                )

# Streamlit UI =================================================================>
def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        login()
    else:
        app()

# Run Streamlit app
if __name__ == "__main__":
    main()
