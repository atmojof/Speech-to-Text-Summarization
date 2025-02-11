import streamlit as st
import time
import whisper
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_community.llms import Ollama
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer

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
    For long files (e.g. more than 5 minutes), the audio is split into segments and each segment is transcribed individually.
    The audio is converted to WAV (16kHz, mono) for compatibility with Azure.
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

    prepared_path = convert_to_wav(audio_file_path)
    
    # Load the prepared WAV file and get its duration (in milliseconds)
    audio = AudioSegment.from_file(prepared_path, format="wav")
    duration_ms = len(audio)
    threshold_ms = 5 * 60 * 1000  # 5 minutes in milliseconds

    # Helper: Transcribe one segment file
    def recognize_segment(segment_path):
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
                offset = result_dict.get("Offset", 0)
                start_seconds = offset / 10**7
                start_minutes = int(start_seconds // 60)
                start_seconds = int(start_seconds % 60)
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
    
    # If the audio duration exceeds the threshold, split and process in segments.
    if duration_ms > threshold_ms:
        print("Audio is long; splitting into segments...")
        segment_transcriptions = []
        for start in range(0, duration_ms, threshold_ms):
            segment = audio[start: start + threshold_ms]
            seg_file = tempfile.mktemp(suffix=".wav")
            segment.export(seg_file, format="wav")
            print(f"Processing segment starting at {start/1000:.0f} seconds...")
            seg_trans = recognize_segment(seg_file)
            segment_transcriptions.append(seg_trans)
            os.remove(seg_file)
        full_transcription = "\n".join(segment_transcriptions)
    else:
        full_transcription = recognize_segment(prepared_path)
    
    # Add a short delay to allow file handles to release before deletion.
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
# Summarize transcription results.
# -----------------------------------------------------------------------------
def summarizer(txt):
    # Define the summarization prompt template.
    # This prompt asks the model to summarize the provided text using bullet points.
    summary_prompt = """
        Summarise the following text using bullet points:
        {text}
        Summary:
    """

    prompt_template = PromptTemplate(template=summary_prompt, input_variables=["text"])

    # Create an LLM instance using your Qwen model via Ollama.
    # Adjust the model identifier as needed; here we assume your model is "qwen:1.8b".
    llm = Ollama(model="qwen:1.8b")

    # Create the LLM chain using the prompt template.
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)

    # Get the summary by running the chain with the transcription text.
    summary = llm_chain.run(text=txt)
    return summary

# -----------------------------------------------------------------------------
# Summarize transcription results V2.
# -----------------------------------------------------------------------------
def summarize_text(text_transcription, additional_context):
    # --- Configuration ---
    API_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    API_URL = f"https://api-inference.huggingface.co/models/{API_MODEL}"
    API_KEY = "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"  # Replace with your API key
    HEADERS = {"Authorization": f"Bearer {API_KEY}"}
    client = InferenceClient(api_key=API_KEY)
    marker = "</think>"
    chunk_size = 13000  # Adjust this as needed

    # --- Helper: Extract summary from model output ---
    def extract_summary(content):
        pattern = rf"{re.escape(marker)}\s*(.*)"
        matches = re.findall(pattern, content, re.DOTALL)
        # Remove markdown bold formatting, if any.
        return re.sub(r'\*\*(.*?)\*\*', r'\1', matches[0]) if matches else content

    # --- Helper: Build prompt messages ---
    def build_prompt(prompt_type, text, num_chunks=None, context=None):
        if prompt_type == "final":
            return [{
                "role": "user",
                "content": (
                    f"The context of following text is {additional_context}.\n\n"
                    f"Summarize the following text in a concise and clear manner.\n\n"
                    f"Return your response in bullet points which covers the key points of the text.\n\n{text}\n"
                    )
                }]
        elif prompt_type == "first":
            return [{
                "role": "user",
                "content": (
                    f"The context of following text is {additional_context}.\n\n"
                    f"Summarize the following text in a concise and clear manner and capture all the key points from this section "
                    f"with a maximum length of {1/num_chunks:.2f} of the original text.\n\n{text}\n"
                    )
                }]
        elif prompt_type == "chunk":
            return [{
                "role": "user",
                "content": (
                    f"Based on the provided context: {context}\n\n"
                    f"Create a concise summary that captures all the key points and main ideas from the following text. "
                    f"Ensure the summary is clear, coherent, and relevant to the context with a maximum length of {1/num_chunks:.2f} of the original text.\n\n{text}\n"
                    )
                }]

    # --- Helper: Query the chat API and extract summary ---
    def query_chat(messages):
        completion = client.chat.completions.create(
            model=API_MODEL,
            messages=messages,
        )
        content = completion.choices[0].message.content
        return extract_summary(content)

    # --- Main Logic ---
    # First, try a simple request to determine if the text is "small"
    payload = {"inputs": text_transcription}
    response = requests.post(API_URL, headers=HEADERS, json=payload)

    if response.status_code == 200:
        # Text is small enough to process in one call.
        return query_chat(build_prompt("final", text_transcription))
    else:
        # Text is too large; split it into chunks.
        summaries = []
        chunks = [text_transcription[i:i + chunk_size] for i in range(0, len(text_transcription), chunk_size)]
        num_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            if i == 0:
                messages = build_prompt("first", chunk, num_chunks=num_chunks)
            else:
                messages = build_prompt("chunk", chunk, num_chunks=num_chunks, context=summaries[-1])
            summaries.append(query_chat(messages))
        # Combine the chunk summaries and run a final summarization.
        combined_summary = " ".join(summaries)
        return query_chat(build_prompt("final", combined_summary))

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

                st.session_state.transcription_time = round(time.time() - start_time, 2)
            
        st.write(f"Processing time: {st.session_state.transcription_time} seconds")
    
    # Display transcription results.
    if st.session_state.transcription_result is not None:
        transcription_text = ""
        for original_filename, transcription in st.session_state.transcription_result.items():
            transcription_text += f"{original_filename}:\n{transcription}\n\n"
        
        st.subheader("Transcription Results")
        st.text_area("", value=transcription_text, height=400, key="transcription_area")
    else:
        st.info("No transcription result available yet.")
    
    # Display summary results if available.
    # ---------------------------------------------------------------- Summarizer -------------------------------------------------------------
    if st.session_state.transcription_result is not None:
        with open('./temp_transcription.txt', "r", encoding="utf-8") as f:
            test_text = f.read()

        start_time = time.time()
        with st.spinner("Summarizing... Please wait...", show_time=True):
            st.session_state.summary_result = summarize_text(text_transcription=test_text, additional_context=st.session_state.context)

        st.subheader("Summary")
        st.write(f"Processing time: {round(time.time() - start_time, 2)} seconds")

        st.text_area(" ", 
                    value=st.session_state.summary_result, 
                    height=300, key="summary_area")

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
    
