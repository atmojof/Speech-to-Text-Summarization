import streamlit as st
import time
import whisper
import tempfile
import os
import numpy as np
from pydub import AudioSegment
import azure.cognitiveservices.speech as speechsdk
import concurrent.futures
from faster_whisper import WhisperModel
from concurrent.futures import ThreadPoolExecutor, as_completed

# FIX
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

# FIX
def preprocess_audio(input_path):
    """
    Preprocess the audio file by converting it to mono and resampling to 16kHz.
    Returns the path to the processed audio file.
    """
    audio = AudioSegment.from_file(input_path) # Load the audio file
    audio = audio.set_channels(1) # Convert to mono
    audio = audio.set_frame_rate(16000) # Resample to 16kHz
    processed_path = tempfile.mktemp(suffix=".wav") # Export the processed audio to a temporary file
    audio.export(processed_path, format="wav")
    return processed_path

# FIX
def transcribe_whisper(files, lang):
    transcription_results = ""
    
    for audio_file in files:
        # Save BytesIO to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_file.write(audio_file.read())
            temp_file_path = temp_file.name
        
        # Preprocess the audio: Convert to mono and resample to 16kHz
        processed_audio_path = preprocess_audio(temp_file_path)
        
        # Split if the file is larger than 25MB
        audio_chunks = split_audio(processed_audio_path, max_size_mb=25)
        
        # Transcribe each chunk
        transcription = ""
        for chunk_path in audio_chunks:
            result = model.transcribe(chunk_path, language=lang, word_timestamps=False)
            for segment in result['segments']:
                start_time = segment['start']
                text = segment['text']
                transcription += f"[{int(start_time // 60):02}:{int(start_time % 60):02}] {text}\n"
        
        # Append the file name and transcription
        transcription_results += f"{audio_file.name}\n{transcription.strip()}\n\n"
    
    return transcription_results.strip()


def transcribe_file_az(audio_file, lang, AZURE_KEY, AZURE_SERVICE_REGION):
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_KEY, region=AZURE_SERVICE_REGION)
    speech_config.speech_recognition_language = lang
    speech_config.output_format = speechsdk.OutputFormat.Detailed

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        temp_file.write(audio_file.read())
        temp_file_path = temp_file.name

    audio_config = speechsdk.audio.AudioConfig(filename=temp_file_path)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    transcriptions = []

    def recognized_callback(evt):
        result_json = evt.result.json
        result_dict = eval(result_json)
        recognized_text = result_dict["DisplayText"]
        offset = result_dict["Offset"]
        start_seconds = offset / 10**7
        start_minutes = int(start_seconds // 60)
        start_seconds = int(start_seconds % 60)
        transcriptions.append(f"[{start_minutes:02}:{start_seconds:02}] {recognized_text}")

    done = False

    def stop_cb(evt):
        nonlocal done
        done = True

    speech_recognizer.recognized.connect(recognized_callback)
    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)

    speech_recognizer.start_continuous_recognition()
    while not done:
        pass
    speech_recognizer.stop_continuous_recognition()

    return "\n".join(transcriptions)

def transcribe_azure(files, lang):
    AZURE_KEY = st.session_state.azure_key
    AZURE_SERVICE_REGION = st.session_state.azure_region

    transcriptions = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(transcribe_file_az, audio_file, lang, AZURE_KEY, AZURE_SERVICE_REGION) for audio_file in files]
        for future in concurrent.futures.as_completed(futures):
            transcriptions.append(future.result())

    return "\n".join(transcriptions)

# ============================================ FASTER WHISPER ===============================================
# -----------------------------------------------------------------------------
# Utility: Prepare audio file (convert if necessary)
# -----------------------------------------------------------------------------
def prepare_audio(file_path, target_ext="mp3", target_channels=1, target_frame_rate=16000, bitrate="64k"):
    """
    Checks if the file has a supported extension (wav or mp3). If it does,
    the file is left as-is. Otherwise, it is converted to the target format (default MP3)
    with the desired channels and sample rate.
    
    Returns the path to the (possibly new) file.
    """
    file_dir, file_name = os.path.split(file_path)
    base, ext = os.path.splitext(file_name)
    ext = ext.lower()
    
    # If file extension is already supported, leave it alone.
    if ext in [".wav", ".mp3"]:
        return file_path

    # Otherwise, load and convert the file using pydub.
    audio = AudioSegment.from_file(file_path)
    
    # Adjust channels and sample rate if needed.
    if audio.channels != target_channels:
        audio = audio.set_channels(target_channels)
    if audio.frame_rate != target_frame_rate:
        audio = audio.set_frame_rate(target_frame_rate)
    
    # Create a temporary file path with the target extension.
    temp_path = tempfile.mktemp(suffix=f".{target_ext}")
    audio.export(temp_path, format=target_ext, bitrate=bitrate)
    print(f"Converted '{file_path}' to '{temp_path}'")
    return temp_path

# -----------------------------------------------------------------------------
# Single-file transcription using Faster Whisper
# -----------------------------------------------------------------------------
def transcribe_single_whisper(audio_path, lang="en", model=None):
    """
    Transcribes a single audio file using Faster Whisper.
    Loads the model with device="cpu" and compute_type="float32".
    """
    if model is None:
        # Note: The "jit" parameter is removed to match the supported constructor arguments.
        model = WhisperModel("small", device="cpu", compute_type="float32")
    
    segments, _ = model.transcribe(audio_path, language=lang)
    transcription = ""
    for segment in segments:
        start_time = segment.start
        text = segment.text
        transcription += f"[{int(start_time // 60):02}:{int(start_time % 60):02}] {text}\n"
    return transcription.strip()

# -----------------------------------------------------------------------------
# Multi-file transcription using Faster Whisper
# -----------------------------------------------------------------------------
def transcribe_faster_whisper(audio_paths, lang="en"):
    """
    Transcribes multiple audio files using Faster Whisper.
    The model is loaded once and then reused for each file.
    Returns a dictionary mapping the original file paths to their transcription.
    """
    # Load the model without the unsupported 'jit' parameter.
    model = WhisperModel("small", device="cpu", compute_type="float32")
    results = {}
    
    for original_path in audio_paths:
        # Prepare the file (convert if necessary)
        prepared_path = prepare_audio(original_path)
        print(f"Transcribing file: {prepared_path}")
        transcription = transcribe_single_whisper(prepared_path, lang, model)
        results[original_path] = transcription
        
        # Remove the temporary file if conversion was done.
        if prepared_path != original_path:
            os.remove(prepared_path)
            
    return results

# -----------------------------------------------------------------------------
# Chooses the transcription method based on the selected model.
# -----------------------------------------------------------------------------
def transcribe_audio(files, language_code, model_choice):
    """
    files: list of file path strings.
    language_code: a list where [0] is for Azure and [1] is for Whisper.
    model_choice: either "Azure AI" or "Whisper".
    """
    if model_choice == "Whisper":
        lang = language_code[1]
        return transcribe_faster_whisper(files, lang)
    elif model_choice == "Azure AI":
        lang = language_code[0]
        return transcribe_azure(files, lang)
    else:
        raise ValueError("Unsupported model choice.")

# -----------------------------------------------------------------------------
# Main Streamlit App
# -----------------------------------------------------------------------------
def main():
    st.title("🔉 Speech-To-Text Summarization")

    # Sidebar for instructions and settings
    with st.sidebar:
        st.header("Instructions")
        st.markdown("""
        1. Upload an audio file using the uploader below. The system will transcribe the audio content.
        2. For audio less than 30 minutes (and in English), choose the Whisper model (relatively faster). 
        3. For audio longer than 30 minutes, use the Azure AI model.
        4. The transcription will be summarized for easier understanding.
        """)
        st.header("Settings")
        st.markdown("""
        - **Transcription Model**: Azure AI / Whisper (small)
        - **Summarization Model**: Gemini 1.5 Flash
        - **API Provider**: Google Generative AI
        """)

    # Initialize session state variables if not already set.
    if "transcription_result" not in st.session_state: st.session_state.transcription_result = None
    if "summary_result" not in st.session_state: st.session_state.summary_result = None
    if "transcription_time" not in st.session_state: st.session_state.transcription_time = 0

    # ============================================== Input ==============================================
    # File uploader
    audio_files = st.file_uploader("Upload audio file", 
                                   type=["wav", "mp3", "ogg", "flac", "aac", "webm", "m4a"], 
                                   accept_multiple_files=True)
    
    if audio_files:
        for aud in audio_files:
            st.audio(aud, format=aud.type)
    
    model_choice = st.selectbox("Choose Model", ("Azure AI", "Whisper"))
    if model_choice == "Azure AI":
        st.session_state.azure_key = st.text_input("Azure API Key")
        st.session_state.azure_region = st.text_input("Azure API Region")
    
    language_options = {
        "Indonesian": ["id-ID", "id"],
        "English": ["en-US", "en"]
    }
    language_choice = st.selectbox("Choose Language", list(language_options.keys()))
    language_code = language_options[language_choice]

    # ============================================== ***** ==============================================
    
    # Submit button: process files if any have been uploaded.
    if audio_files:
        if st.button("Submit"):
            start_time = time.time()
            with st.spinner("Transcribing... Please wait...", show_time=True):
                # Write each uploaded file to a temporary file.
                temp_file_info = []  # List of tuples (original_name, temp_file_path)
                for uploaded_file in audio_files:
                    original_name = uploaded_file.name
                    suffix = os.path.splitext(original_name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.read())
                        temp_file_info.append((original_name, tmp.name))
                
                # Get list of temporary file paths.
                temp_file_paths = [temp_path for _, temp_path in temp_file_info]
                
                # Get transcription results (a dictionary mapping file path to transcription).
                temp_transcription = transcribe_audio(temp_file_paths, language_code, model_choice)
                
                # Map transcription results back to original file names.
                st.session_state.transcription_result = {
                    original_name: temp_transcription[temp_path]
                    for original_name, temp_path in temp_file_info
                }
                
                # Build a formatted transcription text.
                transcription_text = ""
                for original_name, transcription in st.session_state.transcription_result.items():
                    transcription_text += f"Transcription for {original_name}:\n{transcription}\n\n"
                
                # Optionally, save the transcription to a file.
                with open('./temp_transcription.txt', 'w', encoding='utf-8') as f:
                    f.write(transcription_text)
            
            st.session_state.transcription_time = time.time() - start_time
        
        st.write(f"Processing time: {np.round(st.session_state.transcription_time, 2)} seconds")
    
    # Display transcription results.
    if st.session_state.transcription_result is not None:
        transcription_text = ""
        for original_filename, transcription in st.session_state.transcription_result.items():
            transcription_text += f"Transcription for {original_filename}:\n{transcription}\n\n"
        
        st.subheader("Transcription Results")
        st.text_area("", value=transcription_text, height=400, key="transcription_area")
    else:
        st.info("No transcription result available yet.")
    
    # Display summary results if available.
    if st.session_state.summary_result:
        st.subheader("Summary")
        st.text_area("Summary Output", value=st.session_state.summary_result, height=300, key="summary_area")
        st.download_button(
            label="Download Summary",
            data=st.session_state.summary_result,
            file_name="summary.txt",
            mime="text/plain",
            key="download_summary"
        )

            #with st.spinner("Summarizing..."):
            #    st.session_state.summary_result = summarize_text(st.session_state.transcription_result)
            #
            #if st.session_state.summary_result:
            #    st.text_area("Summary", st.session_state.summary_result, height=300)
            #    st.download_button(
            #        label="Download Summary",
            #        data=st.session_state.summary_result,
            #        file_name="summary.txt",
            #        mime="text/plain"
            #    )

## Streamlit UI
#if "logged_in" not in st.session_state:
#    st.session_state.logged_in = False
#if not st.session_state.logged_in:
#    login()
#else:

if __name__ == "__main__":
    main()
    
