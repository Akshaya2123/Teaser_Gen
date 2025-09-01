import os
import json
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from PIL import Image
from faster_whisper import WhisperModel
from transformers import BlipProcessor, BlipForConditionalGeneration
import soundfile as sf
import numpy as np
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

# ------------------ Config ------------------
load_dotenv()

WHISPER_MODEL = "tiny"
IMAGE_CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
DEVICE = "cpu"

#TEASER_TARGET_LENGTH = 30
MAX_TEASER_CLIP_LENGTH = 4.0
SCENE_THRESHOLD = 30.0
MIN_SCENE_LENGTH = 3.0
MAX_SCENE_LENGTH = 20.0
MAX_SCENES = 20
ENSURE_COVERAGE = True
UPSCALE = "1280:720"
BGM_PATH = None

EMOTIONAL_KEYWORDS = [
    "shocking", "reveal", "breaking", "final", "truth", "secret",
    "dramatic", "important", "exclusive", "unbelievable", "surprising"
]

print("Loading models...")
whisper_model = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type="int8")
blip_processor = BlipProcessor.from_pretrained(IMAGE_CAPTION_MODEL)
blip_model = BlipForConditionalGeneration.from_pretrained(IMAGE_CAPTION_MODEL).to(DEVICE)
print("Models loaded.")

# ------------------ Groq LLM Integration ------------------
class GroqLLM:
    def __init__(self, model: str = "llama-3.1-8b-instant", temperature: float = 0.2, max_tokens: int = 800, api_key: str = None):
        from langchain_groq import ChatGroq
        groq_api_key = "your groq api" or os.getenv("GROQ_API_KEY")
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

# ---------------------- Video Helpers -------------------------
def download_video(video_source: str) -> Path:
    output_path = Path("input_video.mp4")
    if output_path.exists():
        return output_path
    if "youtube.com" in video_source or "youtu.be" in video_source:
        subprocess.run(["yt-dlp", "-f", "mp4", "-o", str(output_path), video_source], check=True)
    else:
        local_path = Path(video_source)
        if not local_path.exists():
            raise FileNotFoundError(f"Video not found: {video_source}")
        return local_path
    return output_path

def chunk_video(video_path: Path) -> list:
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

# ---------------------- Clip Analysis -------------------------
def analyze_all_clips(chunks, video_path):
    results = []
    max_workers = 6
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

def analyze_clip(clip_id, video_path, start, end):
    audio_path = f"temp_audio_{clip_id}.wav"
    subprocess.run([
        "ffmpeg","-y","-ss",str(start),"-to",str(end),
        "-i",str(video_path),"-vn","-acodec","pcm_s16le",audio_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    segments, _ = whisper_model.transcribe(audio_path, beam_size=1)
    transcript = " ".join([seg.text for seg in segments]).strip()

    n_words = len(transcript.split())
    speech_density = n_words / max(1e-6, end - start)

    try:
        data, sr = sf.read(audio_path, dtype="float32")
        if data.ndim > 1: data = data.mean(axis=1)
        rms = float(np.sqrt((data**2).mean()))
    except Exception:
        rms = 0.0
    os.remove(audio_path)

    mid_time = (start + end)/2
    frame_path = f"frame_{clip_id}.jpg"
    subprocess.run([
        "ffmpeg","-y","-ss",str(mid_time),"-i",str(video_path),
        "-vframes","1",frame_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    caption = "No visual context"
    if Path(frame_path).exists():
        frame = Image.open(frame_path)
        inputs = blip_processor(images=frame, return_tensors="pt").to(DEVICE)
        out = blip_model.generate(**inputs, max_new_tokens=30)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        os.remove(frame_path)

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
        "transcript": transcript or "No speech detected.",
        "visual_caption": caption,
        "speech_density": speech_density,
        "audio_energy": rms,
        "score_hint": score_hint,
    }

# ---------------------- Selection -----------------------------
def select_clips(video_metadata):
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

# ---------------------- Teaser Assembly -----------------------
def format_srt_time(t: float) -> str:
    h = int(t//3600); m = int((t%3600)//60); s = int(t%60); ms = int((t-int(t))*1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def create_final_teaser(video_path: Path, selected_clips, target_length=30, add_subtitles=True):
    base_teaser = Path("teaser_raw.mp4")
    srt_path = Path("teaser.srt")
    final_output = Path("teaser_final.mp4")

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
        if total >= target_length: break
    if not pruned: pruned = selected_clips[:1]

    ffmpeg_cmd = ["ffmpeg","-y"]
    filter_parts = []
    for i, clip in enumerate(pruned):
        ffmpeg_cmd += ["-ss",str(clip["start_time"]),"-to",str(clip["end_time"]),"-i",str(video_path)]
        filter_parts.append(f"[{i}:v:0][{i}:a:0]")
    filter_complex = "".join(filter_parts)+f"concat=n={len(pruned)}:v=1:a=1[outv][outa]"
    ffmpeg_cmd += ["-filter_complex",filter_complex,"-map","[outv]","-map","[outa]","-c:v","libx264","-preset","veryfast","-crf","23","-c:a","aac","-b:a","128k",str(base_teaser)]
    subprocess.run(ffmpeg_cmd, check=True)

    # regenerate subtitles from teaser itself (ensures sync)
    segments, _ = whisper_model.transcribe(str(base_teaser), beam_size=1)
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(f"{format_srt_time(seg.start)} --> {format_srt_time(seg.end)}\n")
            f.write(seg.text.strip() + "\n\n")

    cmd = [
        "ffmpeg","-y","-i",str(base_teaser),"-vf",f"scale={UPSCALE}:flags=lanczos,subtitles={srt_path}"
    ]
    if add_subtitles:
        cmd = [
            "ffmpeg","-y","-i",str(base_teaser),
            "-vf",f"scale={UPSCALE}:flags=lanczos,subtitles={srt_path}"
        ]
    else:
        cmd = [
            "ffmpeg","-y","-i",str(base_teaser),
            "-vf",f"scale={UPSCALE}:flags=lanczos"
        ]
    
    if BGM_PATH and Path(BGM_PATH).exists():
        cmd += ["-i",BGM_PATH,"-filter_complex","[1:a]volume=0.15[a1];[0:a][a1]amix=inputs=2:duration=first:dropout_transition=2[aout]","-map","0:v","-map","[aout]"]
    cmd += ["-c:v","libx264","-preset","veryfast","-crf","23","-c:a","aac","-b:a","128k","-shortest",str(final_output)]
    subprocess.run(cmd, check=True)

    print("🎬 Final teaser saved to", final_output)
    return final_output

# ---------------------- Main ---------------------------------
def main():
    video_source = "https://youtu.be/Pv0iVoSZzN8?feature=shared"
    video_path = download_video(video_source)
    chunks = chunk_video(video_path)

    metadata = analyze_all_clips(chunks, video_path)
    with open("video_analysis.json","w") as f: json.dump(metadata,f,indent=2)

    selected = select_clips(metadata)
    if not selected:
        print("⚠️ No clips selected. Falling back.")
        selected = metadata[:2]

    teaser = create_final_teaser(video_path, selected, target_length=40)
    print("✅ Teaser ready:", teaser)

    # ----------------- Groq LLM Query -----------------
    with open("video_analysis.json") as f:
        analysis_text = f.read()

    # truncate if too large
    if len(analysis_text.split()) > 3000:
        analysis_text = " ".join(analysis_text.split()[:3000])

    system_prompt = "You are an assistant that answers questions about the video metadata and transcript."
    user_query = "Summarize the key contents and hook of the video"

    llm = GroqLLM()
    response = llm.chat(system_prompt, f"Video analysis JSON:\n{analysis_text}\n\nQuestion: {user_query}")
    print("\n🤖 Groq LLM response:")
    print(response)

def extract_audio(video_path, audio_out=None):
    """
    Extract audio from any video using FFmpeg.
    Returns the path to the extracted audio file.
    """
    if not os.path.exists(video_path):
        return None

    if audio_out is None:
        base = os.path.splitext(video_path)[0]
        audio_out = f"{base}_audio.mp3"

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "mp3", audio_out],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return audio_out
    except subprocess.CalledProcessError:
        return None


if __name__ == "__main__":
    main()
