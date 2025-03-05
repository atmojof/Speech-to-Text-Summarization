import streamlit as st
import time
import tempfile
import os
import json
import re
import requests
import numpy as np
from pydub import AudioSegment
import azure.cognitiveservices.speech as speechsdk
import concurrent.futures
from faster_whisper import WhisperModel
import logging
from google import genai
from google.genai import types

st.set_page_config(initial_sidebar_state="expanded")


# ============================================ AZURE SPEECH SDK ===============================================
# ----------------------------------------------------------------------------- 
# Utility: Prepare audio file (convert if necessary)
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
# Transcribe a single file using Azure Speech-to-Text (with WAV conversion and segmentation)
def transcribe_file_az(audio_file_path, lang, AZURE_KEY, AZURE_SERVICE_REGION):
    """
    Transcribes a single audio file using the Azure Speech SDK.
    For long files (e.g. more than 5 minutes), the audio is split into segments and each segment is transcribed concurrently.
    The audio is converted to WAV (16kHz, mono) for compatibility with Azure.
    
    This version adjusts the timestamps so that each segment’s transcription has an absolute timestamp.
    """
    # Helper to force conversion to WAV (regardless of input extension)
    def convert_to_wav(file_path):
        audio = AudioSegment.from_file(file_path)
        if audio.channels != 1:
            audio = audio.set_channels(1)
        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000)
        temp_wav = tempfile.mktemp(suffix=".wav")
        audio.export(temp_wav, format="wav")
        print(f"Converted '{file_path}' to WAV: '{temp_wav}'")
        return temp_wav

    # Convert the input audio to WAV.
    prepared_path = convert_to_wav(audio_file_path)
    
    # Load the prepared WAV file and get its duration in milliseconds.
    audio = AudioSegment.from_file(prepared_path, format="wav")
    duration_ms = len(audio)
    threshold_ms = 5 * 60 * 1000  # 5 minutes in milliseconds

    # Helper: Transcribe one segment file with a given offset (in milliseconds)
    def recognize_segment(segment_path, chunk_offset_ms):
        speech_config = speechsdk.SpeechConfig(subscription=AZURE_KEY, region=AZURE_SERVICE_REGION)
        speech_config.speech_recognition_language = lang
        speech_config.output_format = speechsdk.OutputFormat.Detailed
        audio_config = speechsdk.audio.AudioConfig(filename=segment_path)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        seg_transcriptions = []
        done = False

        def recognized_callback(evt):
            try:
                result_json = evt.result.json
                result_dict = json.loads(result_json)
                recognized_text = result_dict.get("DisplayText", "")
                offset = result_dict.get("Offset", 0)  # offset in 100-nanosecond units relative to segment start
                # Convert offset to seconds and add the chunk's start (in seconds)
                absolute_seconds = offset / 10**7 + (chunk_offset_ms / 1000)
                start_minutes = int(absolute_seconds // 60)
                start_seconds = int(absolute_seconds % 60)
                seg_transcriptions.append(f"[{start_minutes:02}:{start_seconds:02}] {recognized_text}")
            except Exception as e:
                seg_transcriptions.append(f"Error parsing result: {e}")

        def stop_cb(evt):
            nonlocal done
            done = True

        recognizer.recognized.connect(recognized_callback)
        recognizer.session_stopped.connect(stop_cb)
        recognizer.canceled.connect(stop_cb)

        recognizer.start_continuous_recognition()
        while not done:
            time.sleep(0.1)
        recognizer.stop_continuous_recognition()
        return "\n".join(seg_transcriptions)
    
    # Process the audio either as one piece or split into segments.
    if duration_ms > threshold_ms:
        print("Audio is long; splitting into segments...")
        segments = []
        for start in range(0, duration_ms, threshold_ms):
            segment = audio[start: start + threshold_ms]
            segments.append((start, segment))
        
        # Define a helper to process one segment.
        def process_segment(segment_tuple):
            chunk_offset, seg_audio = segment_tuple
            seg_file = tempfile.mktemp(suffix=".wav")
            seg_audio.export(seg_file, format="wav")
            print(f"Processing segment starting at {chunk_offset/1000:.0f} seconds...")
            seg_trans = recognize_segment(seg_file, chunk_offset)
            try:
                os.remove(seg_file)
            except Exception as e:
                print(f"Error removing segment file {seg_file}: {e}")
            return (chunk_offset, seg_trans)
        
        # Process all segments concurrently.
        with concurrent.futures.ThreadPoolExecutor() as seg_executor:
            futures = [seg_executor.submit(process_segment, seg) for seg in segments]
            results_list = [fut.result() for fut in concurrent.futures.as_completed(futures)]
        
        # Sort segment results by their original start time.
        results_list.sort(key=lambda x: x[0])
        full_transcription = "\n".join([trans for _, trans in results_list])
    else:
        full_transcription = recognize_segment(prepared_path, 0)
    
    # Allow a short delay for file handles to release before deletion.
    time.sleep(0.5)
    # Clean up the prepared WAV file.
    if os.path.exists(prepared_path):
        try:
            os.remove(prepared_path)
        except Exception as e:
            print(f"Could not remove {prepared_path}: {e}")
    
    return full_transcription

# ----------------------------------------------------------------------------- 
# Transcribe multiple files using Azure Speech-to-Text concurrently
def transcribe_azure(files, lang):
    """
    Transcribes multiple files using Azure Speech-to-Text and returns a dictionary
    mapping each file path to its transcription.
    """
    AZURE_KEY = st.session_state.azure_key
    AZURE_SERVICE_REGION = st.session_state.azure_region

    results = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_file = {
            executor.submit(transcribe_file_az, file_path, lang, AZURE_KEY, AZURE_SERVICE_REGION): file_path
            for file_path in files
        }
        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                results[file_path] = future.result()
            except Exception as exc:
                results[file_path] = f"Error during transcription: {exc}"
    return results

# ============================================ FASTER WHISPER ===============================================
# -----------------------------------------------------------------------------
# Single-file transcription using Faster Whisper
# -----------------------------------------------------------------------------
def transcribe_single_whisper(audio_path, lang="en", model=None):
    """
    Transcribes a single audio file using Faster Whisper.
    Loads the model with device="cpu" and compute_type="float32".
    """
    if model is None:
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
# Summarize transcription results V2.
# -----------------------------------------------------------------------------
def summarize_text2(text_transcription,additional_context, lang, GEMINI_API_KEY):
    prompt = (
        f"The context of the following text is: {additional_context}.\n\n"
        f"Provide a summary in bullet points that includes all key information from the following text in {lang}."
        f"Ensure the summary is detailed enough to give a clear understanding of the entire content, including main ideas, important details, and relevant conclusions: {text_transcription}"

    )
    
    client = genai.Client(
        api_key=GEMINI_API_KEY,
    )
  
    model = "gemini-2.0-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=prompt
  
                ),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        response_mime_type="text/plain",
    )
  
    summary = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")
        summary = summary + str(chunk.text)
    response = client.models.generate_content(
      model= model,
      contents=contents,
      config=generate_content_config,
    )
  
    # Extract and print token usage
    token_usage = response.usage_metadata
    #print (token_usage)
    #print("/nToken Usage:")
    #print(f"Prompt Tokens: {token_usage.prompt_token_count}")
    #print(f"Completion Tokens: {token_usage.candidates_token_count}")
    #print(f"Total Tokens: {token_usage.prompt_token_count + token_usage.candidates_token_count}")

    return(summary)
# ============================================ MAIN STREAMLIT APP ===============================================
def main():
    st.title("🔉 Audio Summarizer")
    
    # Sidebar for instructions and settings
    with st.sidebar:
        st.header("Instructions")
        st.markdown("""
        1. Upload an audio file using the uploader below. 
        2. The system will transcribe the audio content.
        3. The transcription will be summarized for easier understanding.
        """)
        st.header("Settings")
        st.markdown("""
        - **Transcription Model**: Azure AI / OpenAI Whisper
        - **Summarization Model**: QwenLM
        """)
        st.header("Notes")
        st.markdown("if summarization fails, try saving the transcription then summarize using **ChatGPT** (or Other AI Tools) with following prompt: Summarize the following text (UPLOAD YOUR TXT FILE).")

    # Initialize session state variables if not already set.
    if "transcription_result" not in st.session_state: st.session_state.transcription_result = None
    if "summary_result" not in st.session_state: st.session_state.summary_result = None
    if "transcription_time" not in st.session_state: st.session_state.transcription_time = 0

    # ---------------------------------------------- Input ----------------------------------------------
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
        #st.session_state.hf_key = st.text_input("Huggingface API Key")
    
    language_options = {
        "Indonesian": ["id-ID", "id"],
        "English": ["en-US", "en"]
    }
    language_choice = st.selectbox("Choose Language", list(language_options.keys()))
    language_code = language_options[language_choice]

    st.session_state.context = st.text_input("Add context about this audio recording (Optional)", placeholder = "e.g. Interview with Client")

    #st.session_state.GEMINI_API_KEY = st.text_input("Gemini API KEY")

    # ---------------------------------------------- Submission ----------------------------------------------
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
                
                # Get transcription results (a dictionary mapping file path to transcription) ---------------------------------------------------------
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

                st.session_state.transcription_time = round(time.time() - start_time, 2)
            
        st.write(f"Processing time: {st.session_state.transcription_time} seconds")
    
    # Display transcription results.
    if st.session_state.transcription_result is not None:
        transcription_text = ""
        for original_filename, transcription in st.session_state.transcription_result.items():
            transcription_text += f"{original_filename}:\n{transcription}\n\n"
        
        st.subheader("Transcription Results")
        st.text_area("Transcription", value=transcription_text, height=400, key="transcription_area")
        st.download_button(
            label="Download Summary",
            data=transcription_text,
            file_name="summary.txt",
            mime="text/plain",
            key="download_transcription"
        )
    else:
        st.info("No transcription result available yet.")
    
    # Display summary results if available.
    # ---------------------------------------------------------------- Summarizer -------------------------------------------------------------
    if st.session_state.transcription_result is not None:
        with open('./temp_transcription.txt', "r", encoding="utf-8") as f:
            test_text = f.read()

        start_time = time.time()
        with st.spinner("Summarizing... Please wait...", show_time=True):
            st.session_state.summary_result = summarize_text2(text_transcription=test_text, additional_context=st.session_state.context, lang=language_options[language_choice][0], GEMINI_API_KEY='AIzaSyDkOs064r7mupvM8yUbX9nMV2vi2r6_1y8')

        st.subheader("Summary")
        st.write(f"Processing time: {round(time.time() - start_time, 2)} seconds")

        st.markdown(st.session_state.summary_result)

        st.download_button(
            label="Download Summary",
            data=st.session_state.summary_result,
            file_name="summary.txt",
            mime="text/plain",
            key="download_summary"
        )
    # ---------------------------------------------------------------- ************** -------------------------------------------------------------

if __name__ == "__main__":
    main()
    