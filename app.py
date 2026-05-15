import subprocess
import base64
import json
import os
import uuid
import threading
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, render_template
from anthropic import Anthropic

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2GB

UPLOAD_DIR = Path("uploads")
CLIPS_DIR = Path("clips")
FRAMES_DIR = Path("temp_frames")
for d in [UPLOAD_DIR, CLIPS_DIR, FRAMES_DIR]:
    d.mkdir(exist_ok=True)

client = Anthropic()

# In-memory job store
jobs: dict[str, dict] = {}

VIBES = {
    "gaming": "Find the most exciting gaming moments: clutch plays, funny deaths, rage moments, epic wins, and hype reactions that would go viral on YouTube/Twitch clips.",
    "funny": "Find the funniest moments: unexpected fails, comedy gold, silly reactions, awkward situations, and anything that would make someone laugh out loud.",
    "emotional": "Find emotionally powerful moments: heartfelt reactions, touching exchanges, dramatic reveals, and scenes that create a strong emotional connection.",
    "viral": "Find moments that would go viral on TikTok or YouTube Shorts: loud reactions, surprising cuts, hype actions, quotable lines, and anything that would stop someone from scrolling.",
}

CLIP_PADDING = 2
FRAME_INTERVAL = 2


def extract_frames(video_path: str, interval: int, job_id: str) -> list[dict]:
    frame_dir = FRAMES_DIR / job_id
    frame_dir.mkdir(exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-i", video_path,
            "-vf", f"fps=1/{interval}",
            "-q:v", "3",
            str(frame_dir / "frame_%04d.jpg"),
            "-y",
        ],
        capture_output=True,
    )
    frames = []
    for fname in sorted(frame_dir.iterdir()):
        if fname.suffix == ".jpg":
            b64 = base64.standard_b64encode(fname.read_bytes()).decode()
            idx = int(fname.stem.split("_")[1]) - 1
            frames.append({"timestamp": idx * interval, "b64": b64})
    return frames


def get_video_duration(video_path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of", "json", video_path,
        ],
        capture_output=True, text=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def find_moments(frames: list[dict], vibe: str, job_id: str) -> list[dict]:
    update_job(job_id, status="analyzing", progress=40, message=f"Sending {len(frames)} frames to Claude...")

    content = [
        {
            "type": "text",
            "text": f"""You are a viral video editor. Analyze these video frames (sampled every {FRAME_INTERVAL}s).

Goal: {VIBES.get(vibe, VIBES['viral'])}

For each exciting moment you find, respond ONLY with a JSON array like:
[
  {{"start": 12, "end": 18, "reason": "huge reaction moment", "title": "crazy_reaction"}},
  {{"start": 45, "end": 52, "reason": "funny fail", "title": "epic_fail"}}
]

Timestamps are in seconds. Be selective — pick only the BEST 3-6 moments.
Respond with ONLY the JSON array, no other text.""",
        }
    ]

    for frame in frames:
        content.append({"type": "text", "text": f"[Frame at {frame['timestamp']}s]"})
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": frame["b64"],
                },
            }
        )

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": content}],
    )

    raw = response.content[0].text.strip()
    return json.loads(raw)


def export_clip(
    video_path: str,
    start: float,
    end: float,
    title: str,
    index: int,
    fmt: str,
    output_dir: Path,
) -> str:
    safe_title = "".join(c if c.isalnum() or c == "_" else "_" for c in title)
    out = str(output_dir / f"clip_{index:02d}_{safe_title}.mp4")

    padded_start = max(0, start - CLIP_PADDING)
    padded_end = end + CLIP_PADDING
    duration = padded_end - padded_start

    if fmt == "vertical":
        vf = "crop=ih*9/16:ih,scale=1080:1920"
    else:
        vf = "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2"

    subprocess.run(
        [
            "ffmpeg", "-i", video_path,
            "-ss", str(padded_start), "-t", str(duration),
            "-vf", vf,
            "-c:v", "libx264", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            out, "-y",
        ],
        capture_output=True,
    )
    return out


def update_job(job_id: str, **kwargs):
    if job_id in jobs:
        jobs[job_id].update(kwargs)


def process_video(job_id: str, video_path: str, vibe: str, fmt: str):
    try:
        update_job(job_id, status="extracting", progress=10, message="Extracting frames from video...")
        frames = extract_frames(video_path, FRAME_INTERVAL, job_id)

        if not frames:
            update_job(job_id, status="error", progress=0, message="Failed to extract frames from video.")
            return

        update_job(job_id, progress=30, message=f"Extracted {len(frames)} frames. Analyzing with Claude...")
        moments = find_moments(frames, vibe, job_id)

        update_job(job_id, status="exporting", progress=60, message=f"Found {len(moments)} moments. Exporting clips...")

        output_dir = CLIPS_DIR / job_id
        output_dir.mkdir(exist_ok=True)

        clips = []
        for i, moment in enumerate(moments):
            pct = 60 + int((i / len(moments)) * 35)
            update_job(job_id, progress=pct, message=f"Exporting clip {i+1} of {len(moments)}: {moment.get('title', 'clip')}...")
            out_path = export_clip(video_path, moment["start"], moment["end"], moment.get("title", f"clip_{i}"), i, fmt, output_dir)
            clips.append(
                {
                    "file": Path(out_path).name,
                    "path": out_path,
                    "reason": moment.get("reason", ""),
                    "title": moment.get("title", f"clip_{i}"),
                    "start": moment["start"],
                    "end": moment["end"],
                    "download_url": f"/clips/{job_id}/{Path(out_path).name}",
                }
            )

        # Clean up frames
        import shutil
        shutil.rmtree(FRAMES_DIR / job_id, ignore_errors=True)

        update_job(job_id, status="done", progress=100, message="All clips exported!", clips=clips)

    except json.JSONDecodeError as e:
        update_job(job_id, status="error", progress=0, message=f"Claude returned unexpected response. Try again or change vibe. ({e})")
    except Exception as e:
        update_job(job_id, status="error", progress=0, message=str(e))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files["video"]
    vibe = request.form.get("vibe", "viral")
    fmt = request.form.get("format", "vertical")

    if not video_file.filename:
        return jsonify({"error": "No file selected"}), 400

    job_id = str(uuid.uuid4())
    ext = Path(video_file.filename).suffix or ".mp4"
    video_path = str(UPLOAD_DIR / f"{job_id}{ext}")
    video_file.save(video_path)

    jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "message": "Job queued...",
        "clips": [],
    }

    thread = threading.Thread(target=process_video, args=(job_id, video_path, vibe, fmt), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/clips/<job_id>/<filename>")
def download_clip(job_id, filename):
    clip_dir = CLIPS_DIR / job_id
    return send_from_directory(str(clip_dir), filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
