# Batch 15 - Department of Computer Science and Engineering (Velammal Engineering College)
## USE CASE - Teaser Creation from Video
## Team Members 
1. Sangamithra V (Team Leader)
2. Shreya S
3. Akshayasree S 
4. Siddharth J S
5. Manimoli A T

## Features

- Upload a video file or paste a YouTube link
- Scene detection and video segmentation
- **Audio transcription** using Whisper  
- **Visual context extraction** using BLIP (or equivalent VLM)  
- Structured **JSON metadata** per clip with timestamps, transcripts, and visual descriptions
- **Sentiment analysis & scoring** to identify emotionally intense moments
- Clip selection guided by **Llama 3.1 LLM** via Groq inference
- Automated teaser assembly with FFmpeg
- Subtitle generation for teasers
- JSON + transcripts export for transparency and further analysis
- **Interactive assistant**: ask questions about the video using the processed JSON as context

---

## Architecture Overview

1. **Frontend**  
   - Built with **Streamlit** (`Front.py`)  
   - Uploads video or accepts YouTube links  
   - Displays teaser outputs and transcripts  

2. **Backend**  
   - Implemented with **FastAPI** (`backend.py`, `app.py`)  
   - Handles video download, preprocessing, clip segmentation, and AI inference  
   - Orchestrates Whisper, BLIP, sentiment scoring, and LLM reasoning  

3. **Processing Pipeline**
   - **Preprocessing**: Split video into scene-based chunks with FFmpeg  
   - **Clip Analysis**:  
     - Whisper → Audio transcripts  
     - BLIP → Visual scene descriptions  
   - **Metadata Storage**: Structured JSON with timestamps, transcripts, descriptions  
   - **Sentiment Scoring**: Evaluate emotional intensity (positive, negative, suspense, shocking)  
   - **LLM Reasoning**: Llama 3.1 (via Groq API) curates best clips for teaser  
   - **Teaser Assembly**: FFmpeg stitches clips + subtitles + optional music  

---

## Tech Stack

- **Python** (core language)
- **FastAPI** – backend services
- **Streamlit** – frontend UI
- **FFmpeg** – video segmentation, stitching, and subtitle embedding
- **Whisper** – audio transcription
- **BLIP** (or equivalent VLM) – visual scene description
- **Transformers / HuggingFace** – NLP models for sentiment analysis
- **Groq LLM API** – inference with Llama 3.1 8B Instant
- **Pydantic + JSON** – structured metadata storage
- **Uvicorn** – ASGI server

---

