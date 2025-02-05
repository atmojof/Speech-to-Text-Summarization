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

@st.cache_data
def split_audio(file_path, max_size_mb=25):
    audio = AudioSegment.from_file(file_path)
    max_size_bytes = max_size_mb * 1024 * 1024
    file_size = os.path.getsize(file_path)

    if file_size <= max_size_bytes:
        return [file_path]

    chunk_duration_ms = (max_size_bytes / file_size) * len(audio)
    chunks = [audio[i:i + int(chunk_duration_ms)] for i in range(0, len(audio), int(chunk_duration_ms))]
    
    chunk_files = []
    for i, chunk in enumerate(chunks):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            chunk.export(temp_file.name, format="wav")
            chunk_files.append(temp_file.name)
    return chunk_files

@st.cache_data
def transcribe_whisper(files, language_code):
    transcription_results = ""

    for audio_file in files:
        file_name = audio_file.name
        file_ext = os.path.splitext(file_name)[1][1:].lower()

        # Convert to MP3 if needed
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_mp3:
            if file_ext != "mp3":
                audio = AudioSegment.from_file(audio_file, format=file_ext)
                audio.export(temp_mp3.name, format="mp3")
            else:
                audio_file.seek(0)
                temp_mp3.write(audio_file.read())
            
            temp_path = temp_mp3.name

        # Split and transcribe
        audio_chunks = split_audio(temp_path)
        transcription = ""
        
        for chunk_path in audio_chunks:
            result = model.transcribe(chunk_path, language=language_code, word_timestamps=False)
            for segment in result['segments']:
                start = segment['start']
                text = segment['text']
                transcription += f"[{int(start//60):02}:{int(start%60):02}] {text}\n"
            os.unlink(chunk_path)
        
        os.unlink(temp_path)
        transcription_results += f"{file_name}\n{transcription.strip()}\n\n"

    return transcription_results.strip()

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

def main():
    st.title("🔉 Audio Transcription")

    # Sidebar for instructions and settings
    with st.sidebar:
        st.header("Instructions")

        st.markdown("""
        1. Upload an audio file using the uploader below.
        2. For Audio < 30 minutes, choose whisper model.
        2. The system will transcribe the audio content.
        3. The transcription will be summarized for easier understanding.
        """)

        st.header("Settings")
        st.markdown("""
        - **Transcription Model**: Azure AI / Whisper (small)
        - **Summarization Model**: Gemini 1.5 Flash
        - **API Provider**: Google Generative AI
        """)

    if "transcription_result" not in st.session_state: st.session_state.transcription_result = None
    if "summary_result" not in st.session_state: st.session_state.summary_result = None

    # Input-1: Model selection (currently only Whisper is implemented)
    model_choice = st.selectbox("Choose Model", ("Azure AI", "Whisper"))
    if model_choice == "Azure AI":
        st.session_state.azure_key = st.text_input("Azure API Key")
    
    # Language selection with correct Whisper codes
    language_options = {
        "id": "Indonesian",
        "en": "English"
    }

    language_choice = st.selectbox("Choose Language", list(language_options.values()))
    language_code = list(language_options.keys())[list(language_options.values()).index(language_choice)]
    SUPPORTED_FORMATS = ["wav", "mp3", "ogg", "flac", "aac", "webm", "m4a"]
    audio_files = st.file_uploader(
        "Upload audio file(s) (Max 25MB each)", 
        type=SUPPORTED_FORMATS, 
        accept_multiple_files=True
    )
    if audio_files and st.button("Submit"):
        start_time = time.time()
        with st.spinner("Transcribing..."):
            st.session_state.transcription_result = transcribe_whisper(audio_files, language_code)
        
        st.write(f"Transcription completed in {time.time()-start_time:.2f} seconds.")
        
        if st.session_state.transcription_result:
            st.subheader("Transcription")
            st.text_area("", st.session_state.transcription_result, height=400)
            st.download_button(
                label="Download Transcription",
                data=st.session_state.transcription_result,
                file_name="transcription.txt",
                mime="text/plain"
            )
            with st.spinner("Summarizing..."):
                st.session_state.summary_result = summarize_text(st.session_state.transcription_result)
            
            if st.session_state.summary_result:
                st.text_area("Summary", st.session_state.summary_result, height=300)
                st.download_button(
                    label="Download Summary",
                    data=st.session_state.summary_result,
                    file_name="summary.txt",
                    mime="text/plain"
                )

# Streamlit UI
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if not st.session_state.logged_in:
    login()
else:
    main()
    
