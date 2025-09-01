from pathlib import Path

def cleanup_temp_files():
    for p in Path(".").glob("temp_audio_*.wav"):
        try:
            p.unlink()
        except:
            pass
    for p in Path(".").glob("frame_*.jpg"):
        try:
            p.unlink()
        except:
            pass
    for f in ["input_video.mp4","teaser_raw.mp4","teaser.srt"]:
        try:
            Path(f).unlink()
        except:
            pass
