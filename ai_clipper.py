# ai_clipper.py
# CLI version: give it a video, it finds the best moments and exports edited clips

import subprocess
import base64
import json
import os
from anthropic import Anthropic

client = Anthropic()

VIDEO_PATH = "input.mp4"
OUTPUT_DIR = "clips/"
FRAME_INTERVAL = 2
CLIP_PADDING = 3

VIBE = """
Find moments that would go viral on TikTok or YouTube Shorts.
Look for: loud reactions, funny moments, hype/exciting actions,
surprising cuts, emotional peaks, or anything that would make
someone stop scrolling.
"""


def extract_frames(video_path, interval=2):
    os.makedirs("temp_frames", exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-i", video_path,
            "-vf", f"fps=1/{interval}",
            "-q:v", "2",
            "temp_frames/frame_%04d.jpg",
            "-y",
        ],
        capture_output=True,
    )

    frames = []
    for fname in sorted(os.listdir("temp_frames")):
        if fname.endswith(".jpg"):
            fpath = f"temp_frames/{fname}"
            with open(fpath, "rb") as f:
                b64 = base64.standard_b64encode(f.read()).decode()
            timestamp = (int(fname.split("_")[1].split(".")[0]) - 1) * interval
            frames.append({"timestamp": timestamp, "b64": b64, "file": fname})
    return frames


def get_video_duration(video_path):
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of", "json", video_path,
        ],
        capture_output=True,
        text=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def find_moments(frames):
    print("🔍 Analyzing video with Claude...")

    content = [
        {
            "type": "text",
            "text": f"""You are a viral video editor. Analyze these video frames (sampled every {FRAME_INTERVAL}s).

Vibe/goal: {VIBE}

For each exciting moment you find, respond ONLY with a JSON array like:
[
  {{"start": 12, "end": 18, "reason": "huge reaction moment", "title": "crazy_reaction"}},
  {{"start": 45, "end": 52, "reason": "funny fail", "title": "epic_fail"}}
]

Each timestamp is in seconds. Be selective — only pick the BEST 3-6 moments.
Respond with ONLY the JSON array, no other text.
""",
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
        max_tokens=1000,
        messages=[{"role": "user", "content": content}],
    )

    raw = response.content[0].text.strip()
    return json.loads(raw)


def export_clip(video_path, start, end, title, index, style="tiktok"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe_title = "".join(c if c.isalnum() or c == "_" else "_" for c in title)
    out = f"{OUTPUT_DIR}clip_{index:02d}_{safe_title}.mp4"

    start = max(0, start - CLIP_PADDING)
    end = end + CLIP_PADDING
    duration = end - start

    if style == "tiktok":
        vf = "crop=ih*9/16:ih,scale=1080:1920:flags=lanczos,unsharp=5:5:0.8:3:3:0.4"
    else:
        vf = ("scale=1920:1080:force_original_aspect_ratio=decrease:flags=lanczos,"
              "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,"
              "unsharp=5:5:0.8:3:3:0.4")

    subprocess.run(
        [
            "ffmpeg", "-i", video_path,
            "-ss", str(start), "-t", str(duration),
            "-vf", vf,
            "-c:v", "libx264", "-crf", "17", "-preset", "slow",
            "-profile:v", "high", "-level", "4.1", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "320k", "-ar", "48000",
            "-movflags", "+faststart",
            out, "-y",
        ],
        capture_output=True,
    )

    print(f"  ✅ Exported: {out}")
    return out


def main():
    print(f"🎬 Loading video: {VIDEO_PATH}")
    duration = get_video_duration(VIDEO_PATH)
    print(f"   Duration: {duration:.1f}s")

    frames = extract_frames(VIDEO_PATH, FRAME_INTERVAL)
    print(f"   Sampled {len(frames)} frames")

    moments = find_moments(frames)
    print(f"\n🎯 Found {len(moments)} moments:")
    for m in moments:
        print(f"   [{m['start']}s → {m['end']}s] {m['reason']}")

    print(f"\n✂️  Exporting clips...")
    for i, moment in enumerate(moments):
        export_clip(
            VIDEO_PATH,
            moment["start"],
            moment["end"],
            moment["title"],
            i,
            style="tiktok",
        )

    print(f"\n🎉 Done! Check the '{OUTPUT_DIR}' folder.")


if __name__ == "__main__":
    main()
