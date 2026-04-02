# app.py
import os
import json
import subprocess
import tempfile
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from dotenv import load_dotenv

# ML / audio / vision libs
try:
    import torch
    from PIL import Image
    import soundfile as sf
    import numpy as np
    from faster_whisper import WhisperModel
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector
except Exception as e:
    # We'll surface this in the UI if models are unavailable
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None

# Load env
load_dotenv()

########### Config (same as backend) ###########
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")
IMAGE_CAPTION_MODEL = os.getenv("IMAGE_CAPTION_MODEL", "Salesforce/blip-image-captioning-base")
DEVICE = os.getenv("DEVICE", "cpu")

#TEASER_TARGET_LENGTH = int(os.getenv("TEASER_TARGET_LENGTH", "30"))
MAX_TEASER_CLIP_LENGTH = float(os.getenv("MAX_TEASER_CLIP_LENGTH", "4.0"))
SCENE_THRESHOLD = float(os.getenv("SCENE_THRESHOLD", "30.0"))
MIN_SCENE_LENGTH = float(os.getenv("MIN_SCENE_LENGTH", "3.0"))
MAX_SCENE_LENGTH = float(os.getenv("MAX_SCENE_LENGTH", "20.0"))
MAX_SCENES = int(os.getenv("MAX_SCENES", "20"))
ENSURE_COVERAGE = True
UPSCALE = os.getenv("UPSCALE", "3840:2160")
BGM_PATH = os.getenv("BGM_PATH", None)

EMOTIONAL_KEYWORDS = [
    "shocking", "reveal", "breaking", "final", "truth", "secret",
    "dramatic", "important", "exclusive", "unbelievable", "surprising"
]

# Frontend constants
SUPPORTED_VIDEO_FORMATS = ["mp4", "mov", "avi", "mkv"]
MAX_FILE_SIZE_MB = 500

########### Model loading (attempt) ###########
whisper_model = None
blip_processor = None
blip_model = None

if _IMPORT_ERROR is None:
    try:
        st_info = f"Loading models: Whisper `{WHISPER_MODEL}` and BLIP `{IMAGE_CAPTION_MODEL}`..."
        print(st_info)
        whisper_model = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type="int8")
        blip_processor = BlipProcessor.from_pretrained(IMAGE_CAPTION_MODEL)
        blip_model = BlipForConditionalGeneration.from_pretrained(IMAGE_CAPTION_MODEL).to(DEVICE)
        print("Models loaded.")
    except Exception as e:
        print("Model loading failed:", e)
        whisper_model = None
        blip_processor = None
        blip_model = None

########### Groq LLM wrapper (optional) ###########
class GroqLLM:
    def __init__(self, model: str = "llama-3.1-8b-instant", temperature: float = 0.2, max_tokens: int = 800, api_key: str = None):
        try:
            from langchain_groq import ChatGroq
        except Exception as e:
            raise RuntimeError("langchain_groq not installed or import failed.") from e

        groq_api_key = "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" or os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise RuntimeError("GROQ_API_KEY missing. Set it in env or pass as argument.")
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def chat(self, system: str, user: str) -> str:
        from langchain_core.messages import SystemMessage, HumanMessage
        msgs = []
        if system:
            msgs.append(SystemMessage(content=system))
        msgs.append(HumanMessage(content=user))
        resp = self.llm.invoke(msgs)
        return str(resp.content).strip().replace("\n", " ")

########### Backend functions (merged) ###########

def download_video(video_source: str) -> Path:
    """
    If video_source is a local path, return Path.
    If it is a youtube URL, download with yt-dlp to input_video.mp4
    """
    output_path = Path("input_video.mp4")
    if output_path.exists():
        return output_path
    if "youtube.com" in video_source or "youtu.be" in video_source:
        cmd = ["yt-dlp", "-f", "mp4", "-o", str(output_path), video_source]
        subprocess.run(cmd, check=True)
        return output_path
    else:
        local_path = Path(video_source)
        if not local_path.exists():
            raise FileNotFoundError(f"Video not found: {video_source}")
        return local_path

def chunk_video(video_path: Path) -> list:
    """
    Use scenedetect to find scenes and return list of (clip_id, start_sec, end_sec)
    """
    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=SCENE_THRESHOLD))
    scene_manager.detect_scenes(video)
    scenes = scene_manager.get_scene_list()

    chunks = []
    clip_id = 1
    for start, end in scenes:
        s, e = start.get_seconds(), end.get_seconds()
        if e - s < MIN_SCENE_LENGTH:
            if chunks:
                prev_id, prev_s, prev_e = chunks[-1]
                chunks[-1] = (prev_id, prev_s, e)
            continue
        while e - s > MAX_SCENE_LENGTH:
            chunks.append((clip_id, s, s + MAX_SCENE_LENGTH))
            clip_id += 1
            s += MAX_SCENE_LENGTH
        chunks.append((clip_id, s, e))
        clip_id += 1

    if len(chunks) > MAX_SCENES:
        head, tail = chunks[:10], chunks[-5:]
        middle = sorted(chunks[10:-5], key=lambda x: (x[2]-x[1]), reverse=True)[:MAX_SCENES-len(head)-len(tail)]
        chunks = head + middle + tail

    print(f"Video split into {len(chunks)} scene-based chunks.")
    return chunks

def analyze_clip(clip_id, video_path, start, end):
    """
    Extract audio for clip, transcribe with WhisperModel, compute audio RMS,
    grab mid-frame and run BLIP captioning. Returns metadata dict for the clip.
    """
    audio_path = f"temp_audio_{clip_id}.wav"
    # extract audio
    subprocess.run([
        "ffmpeg","-y","-ss",str(start),"-to",str(end),
        "-i",str(video_path),"-vn","-acodec","pcm_s16le",audio_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    transcript = "No speech detected."
    n_words = 0
    speech_density = 0.0
    rms = 0.0

    # Transcribe if whisper model available
    if whisper_model:
        try:
            segments, _ = whisper_model.transcribe(audio_path, beam_size=1)
            transcript = " ".join([seg.text for seg in segments]).strip() or transcript
            n_words = len(transcript.split())
            speech_density = n_words / max(1e-6, end - start)
        except Exception as e:
            print("Whisper transcription failed:", e)
    else:
        print("Whisper model not available. Skipping transcription.")

    # audio RMS
    try:
        data, sr = sf.read(audio_path, dtype="float32")
        if data.ndim > 1: data = data.mean(axis=1)
        rms = float(np.sqrt((data**2).mean()))
    except Exception as e:
        print("Audio read failed:", e)
        rms = 0.0

    try:
        os.remove(audio_path)
    except Exception:
        pass

    # extract mid-frame and caption with BLIP if available
    mid_time = (start + end) / 2
    frame_path = f"frame_{clip_id}.jpg"
    subprocess.run([
        "ffmpeg","-y","-ss",str(mid_time),"-i",str(video_path),
        "-vframes","1",frame_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    caption = "No visual context"
    if blip_processor and blip_model and Path(frame_path).exists():
        try:
            frame = Image.open(frame_path).convert("RGB")
            inputs = blip_processor(images=frame, return_tensors="pt").to(DEVICE)
            out = blip_model.generate(**inputs, max_new_tokens=30)
            caption = blip_processor.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            print("BLIP caption failed:", e)
    else:
        print("BLIP unavailable or frame missing; skipping visual caption.")

    try:
        if Path(frame_path).exists():
            os.remove(frame_path)
    except Exception:
        pass

    keyword_boost = sum(0.5 for kw in EMOTIONAL_KEYWORDS if kw in transcript.lower())
    pause_factor = 0.0
    if n_words < 5 and (end - start) > 5:
        pause_factor = 0.3

    score_hint = (
        0.5 * speech_density +
        0.4 * rms +
        0.1 * (len(caption.split())/10) +
        keyword_boost +
        pause_factor
    )

    return {
        "clip_id": clip_id,
        "start_time": start,
        "end_time": end,
        "duration": end - start,
        "transcript": transcript,
        "visual_caption": caption,
        "speech_density": speech_density,
        "audio_energy": rms,
        "score_hint": score_hint,
    }

def analyze_all_clips(chunks, video_path):
    results = []
    max_workers = min(6, max(1, len(chunks)))
    print(f"[Parallel] Using {max_workers} threads for clip analysis...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(analyze_clip, cid, video_path, s, e) for cid, s, e in chunks]
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                print("⚠️ Clip analysis failed:", e)

    results.sort(key=lambda x: x["start_time"])
    return results

def select_clips(video_metadata):
    if not video_metadata:
        return []
    sorted_clips = sorted(video_metadata, key=lambda x: x["score_hint"], reverse=True)
    hook = sorted_clips[0]

    if not ENSURE_COVERAGE:
        return sorted_clips[:8]

    dur = max(c["end_time"] for c in video_metadata)
    intro = [c for c in video_metadata if c["start_time"] < dur*0.33]
    middle = [c for c in video_metadata if dur*0.33 <= c["start_time"] < dur*0.66]
    end = [c for c in video_metadata if c["start_time"] >= dur*0.66]

    def best_clip(clips): return max(clips, key=lambda x: x["score_hint"]) if clips else None

    picks = [hook]
    for pool in (intro, middle, end):
        best = best_clip(pool)
        if best and best["clip_id"] != hook["clip_id"]:
            picks.append(best)

    for c in sorted_clips:
        if c not in picks and len(picks) < 20:
            picks.append(c)

    picks = [picks[0]] + sorted(picks[1:], key=lambda x: x["start_time"])
    return picks

def format_srt_time(t: float) -> str:
    h = int(t//3600); m = int((t%3600)//60); s = int(t%60); ms = int((t-int(t))*1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def create_final_teaser(video_path: Path, selected_clips, output_name: str = "teaser_final.mp4", target_length: int = 30):
    """
    Concatenate selected clips into a teaser, burn subtitles (SRT), optionally mix BGM.
    """
    base_teaser = Path("teaser_raw.mp4")
    srt_path = Path("teaser.srt")
    final_output = Path(output_name)

    # Always sort selected_clips by start_time to preserve original order
    selected_clips = sorted(selected_clips, key=lambda c: c["start_time"])
    total, pruned = 0.0, []
    for clip in selected_clips:
        d = clip["duration"]
        if d > MAX_TEASER_CLIP_LENGTH:
            clip["end_time"] = clip["start_time"] + MAX_TEASER_CLIP_LENGTH
            clip["duration"] = MAX_TEASER_CLIP_LENGTH
            d = MAX_TEASER_CLIP_LENGTH
        if total + d <= target_length + 1:
            pruned.append(clip)
            total += d
        if total >= target_length:
            break
    if not pruned: pruned = selected_clips[:1]

    # Build ffmpeg concat inputs
    ffmpeg_cmd = ["ffmpeg","-y"]
    filter_parts = []
    for i, clip in enumerate(pruned):
        ffmpeg_cmd += ["-ss",str(clip["start_time"]),"-to",str(clip["end_time"]),"-i",str(video_path)]
        filter_parts.append(f"[{i}:v:0][{i}:a:0]")
    filter_complex = "".join(filter_parts) + f"concat=n={len(pruned)}:v=1:a=1[outv][outa]"
    ffmpeg_cmd += ["-filter_complex",filter_complex,"-map","[outv]","-map","[outa]","-c:v","libx264","-preset","veryfast","-crf","23","-c:a","aac","-b:a","128k",str(base_teaser)]
    subprocess.run(ffmpeg_cmd, check=True)

    # write srt
    # regenerate subtitles from teaser itself (ensures sync)
    srt_path = Path("teaser.srt")
    segments, _ = whisper_model.transcribe(str(base_teaser), beam_size=1)
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(f"{format_srt_time(seg.start)} --> {format_srt_time(seg.end)}\n")
            f.write(seg.text.strip() + "\n\n")

    # --- Fade in/out logic ---
    # Get teaser duration
    result = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(base_teaser)
    ], capture_output=True, text=True)
    try:
        teaser_duration = float(result.stdout.strip())
    except Exception:
        teaser_duration = target_length
    fade_dur = min(1.0, teaser_duration * 0.1)
    # Video fade filter
    fade_vf = f"fade=t=in:st=0:d={fade_dur},fade=t=out:st={teaser_duration-fade_dur}:d={fade_dur}"
    # Audio fade filter
    fade_af = f"afade=t=in:ss=0:d={fade_dur},afade=t=out:st={teaser_duration-fade_dur}:d={fade_dur}"
    # Use zscale for 4K upscaling with spline36 filter for best quality
    vf = f"zscale=w=3840:h=2160:filter=spline36,subtitles={srt_path},{fade_vf}"
    cmd = [
        "ffmpeg","-y","-i",str(base_teaser),
        "-vf", vf,
        "-af", fade_af
    ]
    if BGM_PATH and Path(BGM_PATH).exists():
        cmd += ["-i",BGM_PATH,"-filter_complex","[1:a]volume=0.15[a1];[0:a][a1]amix=inputs=2:duration=first:dropout_transition=2[aout]","-map","0:v","-map","[aout]"]
    # Use slow preset and low CRF for best quality
    cmd += ["-c:v","libx264","-preset","slow","-crf","15","-c:a","aac","-b:a","192k","-shortest",str(final_output)]
    subprocess.run(cmd, check=True)

    print("🎬 Final teaser saved to", final_output)
    return final_output

########### Utility helpers ###########
def validate_youtube_url(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url

def generate_caption(tone: str) -> str:
    return f"Here’s a {tone.lower()} teaser for our latest video!"

def cleanup_temp_files():
    for p in Path(".").glob("temp_audio_*.wav"):
        try: p.unlink()
        except: pass
    for p in Path(".").glob("frame_*.jpg"):
        try: p.unlink()
        except: pass
    for f in ["input_video.mp4","teaser_raw.mp4","teaser.srt"]:
        try: Path(f).unlink()
        except: pass

########### Streamlit UI (front) ###########
st.set_page_config(page_title="AI Video Teaser Generator", page_icon="🎬", layout="wide", initial_sidebar_state="collapsed")

def load_css():
    st.markdown("""
    <style>
        * {
            box-sizing: border-box;
        }

        .stApp {
            background: #f8fafc;
            color: #0f172a;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        .block-container {
            max-width: 1120px !important;
            padding-top: 1.6rem !important;
            padding-bottom: 2rem !important;
        }

        .topbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 24px;
            padding: 28px 0 18px;
            border-bottom: 1px solid #e2e8f0;
            margin-bottom: 34px;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 24px;
            font-weight: 900;
            letter-spacing: -0.04em;
            color: #0f172a;
            white-space: nowrap;
            transform: translateY(2px);
        }

        .nav-links {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 30px;
            flex-wrap: wrap;
            font-size: 14px;
            font-weight: 600;
        }

        .nav-links a {
            color: #64748b;
            text-decoration: none;
            padding: 4px 0;
        }

        .nav-links a.active,
        .nav-links a:hover {
            color: #0f172a;
        }

        .nav-right {
            display: flex;
            align-items: center;
            gap: 16px;
            white-space: nowrap;
        }

        .nav-login {
            color: #64748b;
            font-size: 14px;
            font-weight: 500;
        }

        .hero-grid {
            display: grid;
            grid-template-columns: 1.08fr 0.92fr;
            gap: 28px;
            align-items: stretch;
            margin-top: 4px;
        }

        .hero-card,
        .hero-visual,
        .feature-card,
        .panel-card {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }

        .hero-card {
            min-height: 372px;
            padding: 54px 56px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .hero-card h1 {
            font-size: clamp(2.6rem, 5vw, 4rem);
            line-height: 0.98;
            letter-spacing: -0.06em;
            margin: 0 0 22px;
            color: #0f172a;
            max-width: none;
        }

        .hero-card p {
            font-size: 16px;
            line-height: 1.7;
            color: #64748b;
            max-width: 520px;
            margin: 0 0 28px;
        }

        .hero-actions {
            display: flex;
            gap: 14px;
            flex-wrap: wrap;
        }

        .hero-visual {
            min-height: 372px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 36px;
            background: linear-gradient(135deg, rgba(79, 70, 229, 0.08) 0%, rgba(118, 75, 162, 0.08) 100%);
        }

        .hero-visual .icon-shell {
            width: 118px;
            height: 118px;
            border-radius: 28px;
            display: grid;
            place-items: center;
            background: linear-gradient(135deg, rgba(79, 70, 229, 0.14) 0%, rgba(118, 75, 162, 0.2) 100%);
            margin-bottom: 28px;
        }

        .hero-visual .icon-shell span {
            font-size: 58px;
            color: #4f46e5;
        }

        .hero-visual .label {
            color: #334155;
            font-size: 16px;
            font-weight: 700;
            text-align: center;
        }

        .section-title {
            text-align: center;
            font-size: 18px;
            font-weight: 700;
            color: #0f172a;
            margin: 54px 0 26px;
        }

        .feature-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 24px;
            margin-top: 18px;
        }

        .feature-card {
            padding: 26px;
            min-height: 180px;
        }

        .feature-badge {
            width: 42px;
            height: 42px;
            border-radius: 12px;
            display: grid;
            place-items: center;
            background: #eef2ff;
            color: #4f46e5;
            font-size: 20px;
            margin-bottom: 18px;
        }

        .feature-card h3 {
            margin: 0 0 10px;
            font-size: 18px;
            font-weight: 700;
            color: #0f172a;
        }

        .feature-card p {
            margin: 0;
            color: #64748b;
            font-size: 14px;
            line-height: 1.6;
        }

        .panel-card {
            padding: 28px;
            margin-top: 28px;
        }

        .stButton > button {
            border-radius: 10px !important;
            border: 1px solid transparent !important;
            font-weight: 700 !important;
            padding: 0.72rem 1.1rem !important;
            box-shadow: none !important;
        }

        .stButton > button[kind="primary"] {
            background: #4f46e5 !important;
            color: white !important;
        }

        .stButton > button[kind="secondary"] {
            background: white !important;
            color: #0f172a !important;
            border-color: #dbe2ea !important;
        }

        .stButton > button:hover {
            border-color: #cbd5e1 !important;
        }
    </style>
    """, unsafe_allow_html=True)

def init_session_state():
    if "current_step" not in st.session_state: st.session_state.current_step = "welcome"
    if "video_path" not in st.session_state: st.session_state.video_path = None
    if "duration" not in st.session_state: st.session_state.duration = None
    if "tone" not in st.session_state: st.session_state.tone = "Professional"
    if "teaser_path" not in st.session_state: st.session_state.teaser_path = None
    if "caption" not in st.session_state: st.session_state.caption = None
    if "add_subtitles" not in st.session_state: st.session_state.add_subtitles = True
    if "add_music" not in st.session_state: st.session_state.add_music = True
    if "analysis" not in st.session_state: st.session_state.analysis = None

    page = st.query_params.get("page", None)
    if isinstance(page, list):
        page = page[0] if page else None
    route_map = {
        "welcome": "welcome",
        "video_input": "video_input",
        "preferences": "preferences",
        "processing": "processing",
        "output": "output",
    }
    if page in route_map:
        st.session_state.current_step = route_map[page]

def show_top_nav():
    st.markdown(
        """
        <div class="topbar">
            <div class="brand">TeaserGeneration</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_back_row(label, target_step, key):
    left_ratio = 3 if len(label) > 2 else 1
    left_col, _ = st.columns([left_ratio, 12])
    with left_col:
        if st.button(label, key=key, use_container_width=(len(label) <= 2)):
            st.session_state.current_step = target_step
            st.query_params["page"] = target_step
            st.rerun()

def show_welcome():
    left, right = st.columns([1.08, 0.92], gap="large")

    with left:
        st.markdown("""
        <div class="hero-card">
            <h1>Teaser<br>Generation</h1>
            <p>Transform your videos into engaging teaser clips using AI-powered scene detection, intelligent editing, and polished pacing.</p>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="hero-visual">
            <div class="icon-shell"><span>📄</span></div>
            <div class="label">AI-Powered Video Processing</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    primary, secondary = st.columns(2, gap="large")
    with primary:
        if st.button("Get Started", key="start_creation", use_container_width=True):
            st.session_state.current_step = "video_input"
            st.query_params["page"] = "video_input"
            st.rerun()
    with secondary:
        if st.button("See how it works", key="see_workflow", use_container_width=True, type="secondary"):
            st.info("How it works: Start by adding your video, choose your preferred style and duration, then let the app create a short teaser for you. Finally, review the result and download it when you are happy.")

    st.markdown("<div class='section-title'>Why Choose AI Teaser?</div>", unsafe_allow_html=True)
    
    feature_1, feature_2, feature_3 = st.columns(3, gap="large")
    with feature_1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-badge">⚡</div>
            <h3>AI-Powered</h3>
            <p>Smart scene detection automatically identifies the strongest moments from your footage.</p>
        </div>
        """, unsafe_allow_html=True)
    with feature_2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-badge">⏱️</div>
            <h3>Fast Processing</h3>
            <p>Generate professional teaser clips in minutes with a streamlined review flow.</p>
        </div>
        """, unsafe_allow_html=True)
    with feature_3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-badge">🎛️</div>
            <h3>Customizable Output</h3>
            <p>Choose duration, tone, and style to match your brand and audience.</p>
        </div>
        """, unsafe_allow_html=True)


def handle_video_input():
    render_back_row("←", "welcome", "upload_back")
    st.header("Step 1: Provide Your Video")
    input_method = st.radio("Choose input method:", ["Upload a video file", "Paste YouTube URL"], horizontal=True, key="input_method")
    video_source = None

    if input_method == "Upload a video file":
        uploaded_file = st.file_uploader(f"Upload your video ({', '.join(SUPPORTED_VIDEO_FORMATS)})", type=SUPPORTED_VIDEO_FORMATS)
        if uploaded_file is not None:
            if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
                st.error(f"File size exceeds {MAX_FILE_SIZE_MB}MB")
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    video_source = tmp_file.name
                    st.session_state.video_path = video_source
                    st.success("Video uploaded successfully!")
    else:
        youtube_url = st.text_input("Paste YouTube URL:", placeholder="https://www.youtube.com/watch?v=...")
        if youtube_url:
            if validate_youtube_url(youtube_url):
                video_source = youtube_url
                st.session_state.video_path = youtube_url
                st.success("YouTube URL accepted!")
            else:
                st.error("Please enter a valid YouTube URL")

    if video_source:
        if st.button("Continue to Preferences →"):
            st.session_state.current_step = "preferences"
            st.query_params["page"] = "preferences"
            st.rerun()

def get_user_preferences():
    render_back_row("←", "video_input", "prefs_back")
    st.header("Step 2: Teaser Preferences")
    col1 = st.columns(1)[0]
    with col1:
        duration = st.selectbox("Teaser duration:", ["30 seconds", "60 seconds", "Custom"], key="duration_select")
        if duration == "Custom":
            custom_duration = st.slider("Custom duration (seconds):", 10, 120, 30, key="custom_dur")
            st.session_state.duration = custom_duration
        else:
            st.session_state.duration = int(duration.split()[0])
        tone = st.selectbox("Tone:", ["Professional", "Exciting", "Educational", "Inspirational"], key="tone_select")
        st.session_state.tone = tone
        add_subtitles_temp = st.checkbox("Add automatic subtitles", value=st.session_state.add_subtitles, key="add_subs_widget")
    if st.button("Generate Teaser →", key="generate_btn"):
        st.session_state.add_subtitles = add_subtitles_temp
        st.session_state.current_step = "processing"
        st.query_params["page"] = "processing"
        st.rerun()


def process_video():
    render_back_row("←", "preferences", "process_back")
    st.header("Generating Your Teaser")
    if _IMPORT_ERROR is not None:
        st.error(f"Local dependencies missing or failed to import: {_IMPORT_ERROR}")
        st.stop()

    if not st.session_state.video_path:
        st.warning("Please upload a video first.")
        if st.button("Go to Upload", key="process_missing_video"):
            st.session_state.current_step = "video_input"
            st.query_params["page"] = "video_input"
            st.rerun()
        return

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Download if youtube
    if st.session_state.video_path and ("youtube.com" in st.session_state.video_path or "youtu.be" in st.session_state.video_path):
        status_text.text("Downloading YouTube video...")
        try:
            video_path = download_video(st.session_state.video_path)
            st.session_state.video_path = str(video_path)
            progress_bar.progress(10)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            st.error(f"Error downloading YouTube video: {e}")
            st.text(tb)
            print(tb)
            st.session_state.current_step = "video_input"
            st.rerun()
            return
    else:
        video_path = Path(st.session_state.video_path)

    # chunk
    status_text.text("Chunking video into scenes...")
    try:
        chunks = chunk_video(video_path)
        progress_bar.progress(25)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        st.error(f"Scene detection failed: {e}")
        st.text(tb)
        print(tb)
        st.session_state.current_step = "video_input"
        st.rerun()
        return

    # analyze
    status_text.text("Analyzing clips (transcript, audio energy, visual caption)...")
    try:
        metadata = analyze_all_clips(chunks, video_path)
        with open("video_analysis.json", "w") as f:
            json.dump(metadata, f, indent=2)
        st.session_state.analysis = metadata
        progress_bar.progress(65)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        st.error(f"Clip analysis failed: {e}")
        st.text(tb)
        print(tb)
        st.session_state.current_step = "video_input"
        st.rerun()
        return

    # select
    status_text.text("Selecting best clips for teaser...")
    try:
        selected = select_clips(metadata)
        if not selected:
            selected = metadata[:2]
        progress_bar.progress(75)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        st.error(f"Clip selection failed: {e}")
        st.text(tb)
        print(tb)
        st.session_state.current_step = "video_input"
        st.rerun()
        return

    # create teaser
    status_text.text("Creating final teaser (ffmpeg concatenation + subtitles)...")
    try:
        teaser = create_final_teaser(video_path, selected, target_length=st.session_state.duration)
        st.session_state.teaser_path = str(teaser)
        progress_bar.progress(95)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        st.error(f"Teaser creation failed: {e}")
        st.text(tb)
        print(tb)
        st.session_state.current_step = "video_input"
        st.rerun()
        return

    status_text.text("Generating caption...")
    try:
        caption = generate_caption(st.session_state.tone)
        st.session_state.caption = caption
    except Exception:
        st.session_state.caption = "Check out this teaser!"

    progress_bar.progress(100)
    time.sleep(0.5)
    st.session_state.current_step = "output"
    st.query_params["page"] = "output"
    st.rerun()


def show_output_options():
    render_back_row("Home Page", "welcome", "results_home")
    st.header("Your Teaser is Ready!")
    teaser_path = st.session_state.teaser_path

    if teaser_path and os.path.exists(teaser_path):
        st.video(teaser_path)

        action_col_1, action_col_2, action_col_3 = st.columns(3)
        with action_col_1:
            with open(teaser_path, "rb") as f:
                st.download_button(
                    "Download Teaser",
                    data=f.read(),
                    file_name="ai_teaser.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )

        with action_col_2:
            show_analysis = st.button("Show Analysis", use_container_width=True)

        with action_col_3:
            if st.button("Start Over", use_container_width=True):
                cleanup_temp_files()
                for key in list(st.session_state.keys()):
                    if key != "current_step":
                        del st.session_state[key]
                st.session_state.current_step = "welcome"
                st.query_params["page"] = "welcome"
                st.rerun()

        if show_analysis and st.session_state.analysis:
            st.markdown("### Video Analysis")
            for clip in st.session_state.analysis:
                st.write(f"Clip {clip['clip_id']}: {clip['start_time']} - {clip['end_time']}, score {clip['score_hint']:.3f}")
                st.write("Visual caption:", clip.get("visual_caption"))
                st.write("Transcript (excerpt):", (clip.get("transcript") or "")[:500])
                st.markdown("---")
        elif show_analysis:
            st.info("No analysis available.")

    else:
        st.error("Teaser file not found. Please try again.")
        if st.button("Back to start"):
            cleanup_temp_files()
            st.session_state.current_step = "welcome"
            st.query_params["page"] = "welcome"
            st.rerun()

def main():
    load_css()
    init_session_state()
    show_top_nav()

    if st.session_state.current_step == "welcome":
        show_welcome()
    elif st.session_state.current_step == "video_input":
        handle_video_input()
    elif st.session_state.current_step == "preferences":
        get_user_preferences()
    elif st.session_state.current_step == "processing":
        process_video()
    elif st.session_state.current_step == "output":
        show_output_options()

if __name__ == "__main__":
    main()
