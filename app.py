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
UPSCALE = os.getenv("UPSCALE", "1280:720")
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


    cmd = [
        "ffmpeg","-y","-i",str(base_teaser),"-vf",f"scale={UPSCALE}:flags=lanczos,subtitles={srt_path}"
    ]
    if BGM_PATH and Path(BGM_PATH).exists():
        cmd += ["-i",BGM_PATH,"-filter_complex","[1:a]volume=0.15[a1];[0:a][a1]amix=inputs=2:duration=first:dropout_transition=2[aout]","-map","0:v","-map","[aout]"]
    cmd += ["-c:v","libx264","-preset","veryfast","-crf","23","-c:a","aac","-b:a","128k","-shortest",str(final_output)]
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
st.set_page_config(page_title="AI Video Teaser Generator", page_icon="🎬", layout="centered", initial_sidebar_state="collapsed")

def load_css():
    try:
        with open("style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.markdown("""<style>
            .stApp { font-family: 'Inter', sans-serif; background: #ffffff; color: #1a1a1a; }
            .app-title { font-size: 2.4rem; font-weight: 700; color: #1a1a1a; }
        </style>""", unsafe_allow_html=True)

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

def show_welcome():
    st.markdown("<h1 class='app-title'>AI Video Teaser Generator</h1>", unsafe_allow_html=True)
    st.write("Transform your videos into compelling teasers with AI. Upload your content, customize preferences, and create engaging previews in minutes.")
    st.write("---")
    if st.button("Create Your Teaser Now", key="start_creation"):
        st.session_state.current_step = "video_input"
        st.rerun()


def handle_video_input():
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
            st.rerun()

def get_user_preferences():
    st.header("Step 2: Teaser Preferences")
    col1, col2 = st.columns(2)
    with col1:
        duration = st.selectbox("Teaser duration:", ["30 seconds", "60 seconds", "Custom"], key="duration_select")
        if duration == "Custom":
            custom_duration = st.slider("Custom duration (seconds):", 10, 120, 30, key="custom_dur")

            st.session_state.duration = custom_duration
        else:
            st.session_state.duration = int(duration.split()[0])
        tone = st.selectbox("Tone:", ["Professional", "Exciting", "Educational", "Inspirational"], key="tone_select")
        st.session_state.tone = tone
    with col2:
        use_branding = st.checkbox("Add branding elements", key="use_branding")
        if use_branding:
            logo = st.file_uploader("Upload logo (optional):", type=["png","jpg","jpeg"], key="logo_upload")
            tagline = st.text_input("Tagline (optional):", key="tagline_input")
            if logo: st.session_state.logo = logo
            if tagline: st.session_state.tagline = tagline

    add_subtitles_temp = st.checkbox("Add automatic subtitles", value=st.session_state.add_subtitles, key="add_subs_widget")
    add_music_temp = st.checkbox("Add background music", value=st.session_state.add_music, key="add_music_widget")

    if st.button("Generate Teaser →", key="generate_btn"):
        st.session_state.add_subtitles = add_subtitles_temp
        st.session_state.add_music = add_music_temp
        st.session_state.current_step = "processing"
        st.rerun()


def process_video():
    st.header("Generating Your Teaser")
    if _IMPORT_ERROR is not None:
        st.error(f"Local dependencies missing or failed to import: {_IMPORT_ERROR}")
        st.stop()

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
            st.error(f"Error downloading YouTube video: {e}")
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
        st.error(f"Scene detection failed: {e}")
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
        st.error(f"Clip analysis failed: {e}")
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
        st.error(f"Clip selection failed: {e}")
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
        st.error(f"Teaser creation failed: {e}")
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
    st.rerun()


def show_output_options():
    st.header("Your Teaser is Ready!")
    teaser_path = st.session_state.teaser_path
    if teaser_path and os.path.exists(teaser_path):
        st.video(teaser_path)

        st.download_button("Download Teaser", data=open(teaser_path,"rb").read(), file_name="ai_teaser.mp4", mime="video/mp4")
        if st.button("Show video analysis & clips"):
            if st.session_state.analysis:
                for clip in st.session_state.analysis:
                    st.write(f"Clip {clip['clip_id']}: {clip['start_time']} - {clip['end_time']}, score {clip['score_hint']:.3f}")
                    st.write("Visual caption:", clip.get("visual_caption"))
                    st.write("Transcript (excerpt):", (clip.get("transcript") or "")[:500])
                    st.markdown("---")
        st.markdown("### Social caption")
        if st.button("Generate Social Media Caption"):
            st.text_area("Suggested Caption:", st.session_state.caption, height=120)

        

        if st.button("Start Over"):
            cleanup_temp_files()
            for key in list(st.session_state.keys()):
                if key != "current_step":
                    del st.session_state[key]
            st.session_state.current_step = "welcome"
            st.rerun()

    else:
        st.error("Teaser file not found. Please try again.")
        if st.button("Back to start"):
            st.session_state.current_step = "welcome"
            st.rerun()
    st.markdown("### LLM Video Assistant")
    st.write("Ask the assistant about the generated video (summaries, hooks). Requires GROQ_API_KEY in env.")
    llm_input = st.text_input("Ask a question about the video analysis:")
    if st.button("Ask LLM"):
        if not st.session_state.analysis:
            st.error("No analysis present.")
        else:
            try:
                llm = GroqLLM()
                with open("video_analysis.json") as f:
                    analysis_text = f.read()
                # truncate if too large
                if len(analysis_text.split()) > 3000:
                    analysis_text = " ".join(analysis_text.split()[:3000])
                system_prompt = "You are an assistant that answers questions about the video metadata and transcript."
                user_query = f"Video analysis JSON:\n{analysis_text}\n\nQuestion: {llm_input}"
                res = llm.chat(system_prompt, user_query)
                st.success("LLM response:")
                st.write(res)
            except Exception as e:
                st.error(f"LLM query failed: {e}")

def main():
    load_css()
    init_session_state()

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
