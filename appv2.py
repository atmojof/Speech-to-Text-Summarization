import streamlit as st
import whisper
import tempfile
import os
from transformers import pipeline
from pydub import AudioSegment

# Load the Whisper model globally
model = whisper.load_model("base")

# Load the Hugging Face summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Function to split large audio files
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


# Function to summarize text using Hugging Face Transformers
def summarize_text(text):
    try:
        # Automatically adjust max_length based on input size
        max_length = min(150, int(len(text) * 0.3))  # Use 30% of text length for summary length
        min_length = max(30, int(len(text) * 0.1))  # Use 10% of text length for minimum length

        # Split text into chunks if too long for the model
        max_chunk_size = 1024
        text_chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]

        # Summarize each chunk and combine results
        summarized_chunks = summarizer(text_chunks, max_length=max_length, min_length=min_length, do_sample=False)
        summary = " ".join([chunk["summary_text"] for chunk in summarized_chunks])

        return summary
    except Exception as e:
        return f"Error in summarization: {str(e)}"



# Streamlit UI =================================================================>
def main():
    st.title("🔉 Audio Transcription")

    # Initialize session state for transcription
    if "transcription_result" not in st.session_state:
        st.session_state.transcription_result = None
    if "summary_result" not in st.session_state:
        st.session_state.summary_result = None

    # Input-1: Model selection (currently only Whisper is implemented)
    model_choice = st.selectbox("Choose Model", ("Whisper", "Azure AI", "Google Gemini"))

    # Input-2: File uploader
    SUPPORTED_FORMATS = ["wav", "mp3", "ogg", "flac", "aac", "webm", "m4a"]
    audio_files = st.file_uploader(
        "Upload audio file(s) (Max 25MB each)", 
        type=SUPPORTED_FORMATS, 
        accept_multiple_files=True
    )

    # Submit
    if audio_files: 
        #st.audio(audio_file, format=audio_file.type)
        submitted = st.button("Submit")

        if submitted:
            with st.spinner("Transcribing..."):
                # Transcribe the audio file and save it in session state
                st.session_state.transcription_result = transcribe_whisper(audio_files)
                
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

    


# Run Streamlit app
if __name__ == "__main__":
    main()
