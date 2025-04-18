# Audio Summarizer  

## 📌 Overview  

The **Audio Summarizer** is a web-based application that allows users to upload audio files, automatically transcribe them, and generate concise summaries of the audio content. It supports multiple audio formats and leverages advanced AI models for transcription and summarization.  

---

## ✨ Key Features  

✅ **Multi-format Support** – Works with WAV, MP3, OGG, FLAC, AAC, WEBM, and M4A.  
✅ **Multi-model Transcription** – Choose between:  
   - **Azure Speech-to-Text** (requires API key)  
   - **Whisper** (open-source)  
✅ **AI-Powered Summarization** – Uses **Google Gemini AI** to generate summaries.  
✅ **Multi-language Support** – Supports **dozens of languages** for transcription.  
✅ **Export Options** – Download results as **PDF** with formatted text.  

---

## 🛠️ Technologies Used  

- **Python** (Backend)  
- **Streamlit** (Frontend)  
- **Azure Cognitive Services** (Speech-to-Text)  
- **Faster Whisper** (Open-source transcription)  
- **Google Gemini AI** (Summarization)  
- **PyDub** (Audio processing)  
- **xhtml2pdf** (PDF generation)  

---

## 🚀 How to Use  

### **1. Installation**  
Clone the repository and install dependencies:  
```bash
git clone [repo-url]
cd audio-summarizer
pip install -r requirements.txt
```

### **2. Running the App**  
Start the Streamlit application:  
```bash
streamlit run app.py
```

### **3. Usage Guide**  
1. **Upload** one or multiple audio files.  
2. **Select a transcription model**:  
   - **Azure AI** (requires API key and region)  
   - **Whisper** (open-source, no API needed)  
3. **Choose the audio language**.  
4. *(Optional)* Add context about the recording (e.g., "Interview about Project X").  
5. Click **"Submit"** to start processing.  
6. View the **transcription** and **AI-generated summary**.  
7. Download results as a **PDF** with formatted text.  

---

## 📂 File Structure  
```
audio-summarizer/  
├── app.py               # Main Streamlit application  
├── README.md            # Documentation  
├── requirements.txt     # Python dependencies  
└── temp_transcription.txt  # Temporary transcription storage  
```

---

## 🔧 Troubleshooting  

- **Azure API Errors**: Ensure your API key and region are correct.  
- **Whisper Performance**: For long audio files, processing may take time.  
- **Gemini AI Issues**: If summarization fails, try saving the transcription and summarizing manually with ChatGPT.  

---

## 📜 License  
This project is open-source under the **MIT License**.  

---

## 🙌 Contributing  
Contributions are welcome! Feel free to open issues or submit pull requests.  

🚀 **Happy Summarizing!** 🎧📝
