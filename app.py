import subprocess, base64, json, os, uuid, threading, re, shutil, time
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, render_template
from anthropic import Anthropic

try:
    import whisper as openai_whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    import yt_dlp
    YTDLP_AVAILABLE = True
except ImportError:
    YTDLP_AVAILABLE = False

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024 * 1024

UPLOAD_DIR   = Path("uploads")
CLIPS_DIR    = Path("clips")
FRAMES_DIR   = Path("temp_frames")
MUSIC_DIR    = Path("static/music")
HISTORY_FILE = Path("history.json")
FONT_PATH    = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"

for d in [UPLOAD_DIR, CLIPS_DIR, FRAMES_DIR, MUSIC_DIR]:
    d.mkdir(parents=True, exist_ok=True)

client = Anthropic()
jobs:    dict[str, dict] = {}
dl_jobs: dict[str, dict] = {}
history_lock = threading.Lock()
_whisper_model = None

CLIP_PADDING   = 2
FRAME_INTERVAL = 2
VALID_TAGS     = ["Hype", "Funny", "Emotional", "Quotable", "Surprising", "Dramatic", "Wholesome", "Awkward"]

PRESETS = {
    "tiktok":  {"w": 1080, "h": 1920, "label": "TikTok 9:16"},
    "shorts":  {"w": 1080, "h": 1920, "label": "YouTube Shorts 9:16"},
    "reels":   {"w": 1080, "h": 1920, "label": "Instagram Reels 9:16"},
    "youtube": {"w": 1920, "h": 1080, "label": "YouTube 16:9"},
}

VIBES = {
    "gaming":   "Find gaming highlights: clutch plays, epic wins, rage moments, hype reactions that would go viral.",
    "funny":    "Find the funniest moments: fails, unexpected comedy, silly reactions, awkward situations.",
    "emotional":"Find emotionally powerful moments: heartfelt reactions, touching exchanges, dramatic reveals.",
    "viral":    "Find moments that would go viral on TikTok/Shorts: loud reactions, surprising cuts, hype actions, quotable lines.",
    "custom":   "",
}

# ---------------------------------------------------------------------------
# Startup: generate background music tracks with FFmpeg
# ---------------------------------------------------------------------------

def _gen_music():
    tracks = {
        "lofi":     ("aevalsrc=0.15*sin(2*PI*110*t)+0.1*sin(2*PI*165*t)+0.08*sin(2*PI*220*t):s=44100",
                     "volume=0.5,lowpass=f=1800"),
        "hype":     ("aevalsrc=0.18*sin(2*PI*220*t)+0.12*sin(2*PI*330*t)+0.08*sin(2*PI*440*t):s=44100",
                     "volume=0.5"),
        "chill":    ("anoisesrc=color=brown:amplitude=0.04:s=44100",
                     "lowpass=f=500,volume=3"),
        "dramatic": ("aevalsrc=0.15*sin(2*PI*146*t)+0.1*sin(2*PI*175*t)+0.08*sin(2*PI*220*t):s=44100",
                     "volume=0.5,lowpass=f=2500"),
    }
    for name, (src, af) in tracks.items():
        fp = MUSIC_DIR / f"{name}.mp3"
        if not fp.exists():
            subprocess.run(["ffmpeg", "-f", "lavfi", "-i", src,
                            "-t", "120", "-af", af, str(fp), "-y"],
                           capture_output=True)

threading.Thread(target=_gen_music, daemon=True).start()

# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

def load_history() -> dict:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text())
        except Exception:
            pass
    return {"sessions": []}

def append_history(session: dict):
    with history_lock:
        data = load_history()
        data["sessions"].insert(0, session)
        data["sessions"] = data["sessions"][:100]
        HISTORY_FILE.write_text(json.dumps(data, indent=2))

# ---------------------------------------------------------------------------
# Whisper
# ---------------------------------------------------------------------------

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None and WHISPER_AVAILABLE:
        _whisper_model = openai_whisper.load_model("base")
    return _whisper_model

def transcribe(video_path: str) -> dict | None:
    model = get_whisper_model()
    if model is None:
        return None
    try:
        result = model.transcribe(video_path, word_timestamps=True, verbose=False)
        return result
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------

def extract_frames(video_path: str, interval: int, job_id: str) -> list[dict]:
    frame_dir = FRAMES_DIR / job_id
    frame_dir.mkdir(exist_ok=True)
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vf", f"fps=1/{interval}", "-q:v", "3",
        str(frame_dir / "frame_%04d.jpg"), "-y",
    ], capture_output=True)
    frames = []
    for fname in sorted(frame_dir.iterdir()):
        if fname.suffix == ".jpg":
            b64 = base64.standard_b64encode(fname.read_bytes()).decode()
            idx = int(fname.stem.split("_")[1]) - 1
            frames.append({"timestamp": idx * interval, "b64": b64})
    return frames

def get_video_duration(video_path: str) -> float:
    r = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", video_path,
    ], capture_output=True, text=True)
    return float(json.loads(r.stdout)["format"]["duration"])

# ---------------------------------------------------------------------------
# Claude analysis
# ---------------------------------------------------------------------------

CLAUDE_SYSTEM = """You are a viral video editor AI. You receive sampled video frames and optionally a transcript.
Return ONLY a valid JSON array of clip objects — no markdown, no extra text.
Each object must have exactly these keys:
  "start"       : float  (seconds)
  "end"         : float  (seconds)
  "peak_moment" : float  (seconds, the single most intense instant, for zoom targeting)
  "title"       : string (snake_case, no spaces, ≤ 30 chars)
  "reason"      : string (1 sentence explaining why this clip is great)
  "tags"        : array  (1-3 of: Hype Funny Emotional Quotable Surprising Dramatic Wholesome Awkward)

Pick the BEST 3-6 moments. Be selective. Timestamps are in seconds from video start."""

def find_moments(frames: list[dict], vibe_text: str, transcript: str | None,
                 exclude_ranges: list[tuple] | None, job_id: str) -> list[dict]:
    update_job(job_id, status="analyzing", progress=50,
               message=f"Sending {len(frames)} frames to Claude…")

    transcript_block = ""
    if transcript:
        transcript_block = f"\n\nFULL TRANSCRIPT:\n{transcript[:6000]}"

    exclude_block = ""
    if exclude_ranges:
        pairs = ", ".join(f"{s:.0f}s-{e:.0f}s" for s, e in exclude_ranges)
        exclude_block = f"\n\nDO NOT include moments already selected: {pairs}"

    content: list[dict] = [{
        "type": "text",
        "text": (f"Goal: {vibe_text}"
                 f"{transcript_block}"
                 f"{exclude_block}"
                 f"\n\nFrames sampled every {FRAME_INTERVAL}s:"),
    }]
    for frame in frames:
        content.append({"type": "text", "text": f"[t={frame['timestamp']}s]"})
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/jpeg", "data": frame["b64"],
        }})

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2048,
        system=CLAUDE_SYSTEM,
        messages=[{"role": "user", "content": content}],
    )
    raw = response.content[0].text.strip()
    # Strip markdown code fences if present
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)

# ---------------------------------------------------------------------------
# Conversational re-analysis
# ---------------------------------------------------------------------------

def chat_with_claude(job_id: str, user_message: str) -> dict:
    job = jobs.get(job_id, {})
    transcript = job.get("transcript_text", "")
    current_clips = job.get("clips", [])
    conversation = job.get("conversation", [])

    clips_summary = "\n".join(
        f"Clip {i+1}: [{c['start']}s-{c['end']}s] '{c['title']}' — {c['reason']} (tags: {', '.join(c.get('tags',[]))})"
        for i, c in enumerate(current_clips)
    )

    system = (
        "You are an AI video editor assistant. The user has already analyzed a video.\n"
        "If the user asks for new/different clips, respond with ONLY a JSON array using the same schema as before.\n"
        "If the user is asking a question or making a comment, respond with a short plain text answer.\n"
        "Never mix JSON and text in the same response.\n\n"
        f"Video transcript (first 4000 chars):\n{transcript[:4000]}\n\n"
        f"Currently selected clips:\n{clips_summary}"
    )

    messages = list(conversation) + [{"role": "user", "content": user_message}]

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2048,
        system=system,
        messages=messages,
    )
    reply = response.content[0].text.strip()

    # Detect if reply is JSON
    stripped = re.sub(r"^```[a-z]*\n?", "", reply)
    stripped = re.sub(r"\n?```$", "", stripped).strip()
    try:
        new_clips_raw = json.loads(stripped)
        if isinstance(new_clips_raw, list):
            return {"type": "clips", "clips": new_clips_raw, "reply": "Found new clips based on your request."}
    except (json.JSONDecodeError, ValueError):
        pass

    return {"type": "text", "reply": reply}

# ---------------------------------------------------------------------------
# Caption generation (ASS format)
# ---------------------------------------------------------------------------

def _ass_time(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h}:{m:02d}:{sec:05.2f}"

def generate_ass(whisper_result: dict, clip_start: float, clip_end: float,
                 path: str, vertical: bool = True):
    w, h = (1080, 1920) if vertical else (1920, 1080)
    fs   = 78 if vertical else 56
    mv   = 280 if vertical else 90
    duration = clip_end - clip_start

    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {w}\nPlayResY: {h}\nWrapStyle: 1\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, Strikeout, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,Arial Black,{fs},&H00FFFFFF,&H000000FF,&H00000000,"
        f"&H80000000,-1,0,0,0,100,100,0,0,1,5,2,2,40,40,{mv},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    events = []
    segments = whisper_result.get("segments", [])

    # Collect all words in clip range
    all_words = []
    for seg in segments:
        for w_obj in seg.get("words", []):
            ws = w_obj.get("start", 0)
            we = w_obj.get("end", 0)
            word = w_obj.get("word", "").strip()
            if not word:
                continue
            if we > clip_start and ws < clip_end:
                all_words.append({"word": word, "start": ws - clip_start, "end": we - clip_start})

    if all_words:
        chunk = 4
        for i in range(0, len(all_words), chunk):
            group = all_words[i:i + chunk]
            t0 = max(0.0, group[0]["start"])
            t1 = min(duration, group[-1]["end"])
            if t0 >= duration:
                continue
            text = " ".join(w["word"].upper() for w in group)
            text = text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
            events.append(f"Dialogue: 0,{_ass_time(t0)},{_ass_time(t1)},Default,,0,0,0,,{text}")
    else:
        for seg in segments:
            ss, se = seg["start"], seg["end"]
            if se <= clip_start or ss >= clip_end:
                continue
            t0 = max(0.0, ss - clip_start)
            t1 = min(duration, se - clip_start)
            text = seg["text"].strip().upper()
            text = text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
            events.append(f"Dialogue: 0,{_ass_time(t0)},{_ass_time(t1)},Default,,0,0,0,,{text}")

    Path(path).write_text(header + "\n".join(events), encoding="utf-8")

# ---------------------------------------------------------------------------
# Clip export (two-pass: format+zoom → captions+music)
# ---------------------------------------------------------------------------

def export_clip(video_path: str, start: float, end: float, title: str, index: int,
                preset: str, output_dir: Path,
                whisper_result: dict | None = None,
                music_track: str | None = None,
                enable_zoom: bool = True,
                enable_captions: bool = True) -> str:

    safe = re.sub(r"[^a-zA-Z0-9_]", "_", title)[:40]
    cfg  = PRESETS.get(preset, PRESETS["tiktok"])
    w, h = cfg["w"], cfg["h"]
    vertical = (h > w)

    padded_start = max(0.0, start - CLIP_PADDING)
    padded_end   = end + CLIP_PADDING
    duration     = padded_end - padded_start
    peak_in_clip = (start - padded_start) + (end - start) / 2.0

    temp_path  = str(output_dir / f"_tmp_{index}.mp4")
    final_path = str(output_dir / f"clip_{index:02d}_{safe}.mp4")

    # ── Pass 1: cut + format + optional zoom ──────────────────────────────
    if vertical:
        crop_vf = f"crop=ih*9/16:ih,scale={w}:{h}"
    else:
        crop_vf = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                   f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2")

    if enable_zoom:
        fps = 30
        zi = max(0, int((peak_in_clip - 0.5) * fps))
        zo = int((peak_in_clip + 0.6) * fps)
        zexpr = (f"if(between(on,{zi},{zo}),"
                 f"min(zoom+0.005,1.1),"
                 f"max(zoom-0.005,1))")
        zoom_vf = (f"zoompan=z='{zexpr}'"
                   f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
                   f":d=1:s={w}x{h}:fps={fps}")
        vf1 = f"{crop_vf},fps={fps},{zoom_vf}"
    else:
        vf1 = crop_vf

    subprocess.run([
        "ffmpeg", "-ss", str(padded_start), "-t", str(duration),
        "-i", video_path,
        "-vf", vf1,
        "-c:v", "libx264", "-crf", "20", "-r", "30",
        "-c:a", "aac", "-b:a", "192k",
        temp_path, "-y",
    ], capture_output=True)

    if not Path(temp_path).exists():
        return ""

    # ── Pass 2: captions + music ──────────────────────────────────────────
    cap_file = None
    if enable_captions and whisper_result:
        cap_file = str(output_dir / f"_cap_{index}.ass")
        generate_ass(whisper_result, padded_start, padded_end, cap_file, vertical)

    music_path = None
    if music_track and music_track != "none":
        candidate = MUSIC_DIR / f"{music_track}.mp3"
        if candidate.exists():
            music_path = str(candidate)

    has_cap   = cap_file and Path(cap_file).exists()
    has_music = bool(music_path)

    if not has_cap and not has_music:
        Path(temp_path).rename(final_path)
        return final_path

    inputs = ["ffmpeg", "-i", temp_path]
    if has_music:
        inputs += ["-i", music_path]

    if has_cap and has_music:
        # Escape colon in path for FFmpeg subtitles filter (Linux is fine, just ensure no spaces)
        cap_esc = cap_file.replace("\\", "/")
        fc = (f"[0:v]subtitles={cap_esc}[vout];"
              f"[1:a]atrim=0:{duration:.2f},asetpts=PTS-STARTPTS,volume=0.18[mus];"
              f"[0:a][mus]amix=inputs=2:duration=first[aout]")
        cmd2 = inputs + ["-filter_complex", fc,
                         "-map", "[vout]", "-map", "[aout]",
                         "-c:v", "libx264", "-crf", "20",
                         "-c:a", "aac", final_path, "-y"]
    elif has_cap:
        cap_esc = cap_file.replace("\\", "/")
        cmd2 = inputs + ["-vf", f"subtitles={cap_esc}",
                         "-c:v", "libx264", "-crf", "20",
                         "-c:a", "copy", final_path, "-y"]
    else:  # has_music only
        fc = (f"[1:a]atrim=0:{duration:.2f},asetpts=PTS-STARTPTS,volume=0.18[mus];"
              f"[0:a][mus]amix=inputs=2:duration=first[aout]")
        cmd2 = inputs + ["-filter_complex", fc,
                         "-map", "0:v", "-map", "[aout]",
                         "-c:v", "copy", "-c:a", "aac", final_path, "-y"]

    result = subprocess.run(cmd2, capture_output=True)

    # If subtitles filter failed, retry without captions
    if result.returncode != 0 and has_cap:
        if has_music:
            fc = (f"[1:a]atrim=0:{duration:.2f},asetpts=PTS-STARTPTS,volume=0.18[mus];"
                  f"[0:a][mus]amix=inputs=2:duration=first[aout]")
            cmd3 = ["ffmpeg", "-i", temp_path, "-i", music_path,
                    "-filter_complex", fc,
                    "-map", "0:v", "-map", "[aout]",
                    "-c:v", "copy", "-c:a", "aac", final_path, "-y"]
        else:
            cmd3 = ["ffmpeg", "-i", temp_path, "-c", "copy", final_path, "-y"]
        subprocess.run(cmd3, capture_output=True)

    Path(temp_path).unlink(missing_ok=True)
    if cap_file:
        Path(cap_file).unlink(missing_ok=True)

    return final_path if Path(final_path).exists() else ""

# ---------------------------------------------------------------------------
# Job helpers
# ---------------------------------------------------------------------------

def update_job(job_id: str, **kwargs):
    if job_id in jobs:
        jobs[job_id].update(kwargs)

def update_dl(job_id: str, **kwargs):
    if job_id in dl_jobs:
        dl_jobs[job_id].update(kwargs)

# ---------------------------------------------------------------------------
# Main processing worker
# ---------------------------------------------------------------------------

def process_video(job_id: str, video_path: str, vibe: str, custom_vibe: str,
                  preset: str, music_track: str,
                  enable_captions: bool, enable_zoom: bool,
                  exclude_ranges: list | None = None):
    try:
        # 1. Extract frames
        update_job(job_id, status="extracting", progress=5,
                   message="Extracting frames…")
        frames = extract_frames(video_path, FRAME_INTERVAL, job_id)
        if not frames:
            update_job(job_id, status="error", message="Could not extract frames.")
            return

        # 2. Transcribe
        whisper_result = None
        transcript_text = ""
        if WHISPER_AVAILABLE:
            update_job(job_id, progress=20, message="Transcribing audio with Whisper…")
            whisper_result = transcribe(video_path)
            if whisper_result:
                transcript_text = whisper_result.get("text", "")

        # 3. Determine vibe prompt
        if vibe == "custom" and custom_vibe.strip():
            vibe_text = custom_vibe.strip()
        else:
            vibe_text = VIBES.get(vibe, VIBES["viral"])

        # 4. Claude analysis
        update_job(job_id, progress=40, message=f"Analyzing {len(frames)} frames with Claude…")
        moments = find_moments(frames, vibe_text, transcript_text or None,
                               exclude_ranges, job_id)

        shutil.rmtree(FRAMES_DIR / job_id, ignore_errors=True)

        # 5. Export clips
        update_job(job_id, status="exporting", progress=60,
                   message=f"Found {len(moments)} moments. Exporting…")

        output_dir = CLIPS_DIR / job_id
        output_dir.mkdir(exist_ok=True)

        clips = []
        for i, m in enumerate(moments):
            pct = 60 + int((i / len(moments)) * 35)
            update_job(job_id, progress=pct,
                       message=f"Exporting clip {i+1}/{len(moments)}: {m.get('title','clip')}…")
            out = export_clip(
                video_path,
                float(m["start"]), float(m["end"]),
                m.get("title", f"clip_{i}"), i, preset, output_dir,
                whisper_result=whisper_result,
                music_track=music_track,
                enable_zoom=enable_zoom,
                enable_captions=enable_captions,
            )
            if not out:
                continue
            clips.append({
                "file":         Path(out).name,
                "title":        m.get("title", f"clip_{i}"),
                "reason":       m.get("reason", ""),
                "tags":         m.get("tags", []),
                "start":        float(m["start"]),
                "end":          float(m["end"]),
                "peak_moment":  float(m.get("peak_moment", (m["start"] + m["end"]) / 2)),
                "download_url": f"/clips/{job_id}/{Path(out).name}",
                "preview_url":  f"/preview/{job_id}/{Path(out).name}",
            })

        # Save conversation seed + history
        jobs[job_id].update({
            "status": "done", "progress": 100,
            "message": "All clips exported!",
            "clips": clips,
            "transcript_text": transcript_text,
            "video_path": video_path,
            "conversation": [],
        })

        append_history({
            "id":          job_id,
            "created_at":  datetime.utcnow().isoformat(),
            "video_name":  Path(video_path).name,
            "vibe":        vibe,
            "custom_vibe": custom_vibe,
            "preset":      preset,
            "clip_count":  len(clips),
            "clips":       clips,
        })

    except json.JSONDecodeError as e:
        shutil.rmtree(FRAMES_DIR / job_id, ignore_errors=True)
        update_job(job_id, status="error", progress=0,
                   message=f"Claude returned unexpected format — try again. ({e})")
    except Exception as e:
        shutil.rmtree(FRAMES_DIR / job_id, ignore_errors=True)
        update_job(job_id, status="error", progress=0, message=str(e))

# ---------------------------------------------------------------------------
# yt-dlp download worker
# ---------------------------------------------------------------------------

def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)

def run_download(job_id: str, url: str):
    downloaded_path: list[str] = []

    def progress_hook(d):
        if d["status"] == "downloading":
            raw = _strip_ansi(d.get("_percent_str", "0%")).replace("%", "").strip()
            try:
                pct = min(95, int(float(raw)))
            except ValueError:
                pct = dl_jobs[job_id].get("progress", 0)
            speed = _strip_ansi(d.get("_speed_str", "")).strip()
            eta   = _strip_ansi(d.get("_eta_str", "")).strip()
            msg   = f"Downloading… {pct}%"
            if speed: msg += f"  {speed}"
            if eta and eta not in ("Unknown", "--:--"): msg += f"  ETA {eta}"
            update_dl(job_id, progress=pct, message=msg)
        elif d["status"] == "finished":
            update_dl(job_id, progress=96, message="Finalising…")
            downloaded_path.append(d.get("filename", ""))

    outtmpl = str(UPLOAD_DIR / "%(title).80s.%(ext)s")
    ydl_opts = {
        "format":             "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
        "merge_output_format":"mp4",
        "outtmpl":            outtmpl,
        "progress_hooks":     [progress_hook],
        "quiet":              True,
        "no_warnings":        True,
        "noplaylist":         True,
        "postprocessors": [{
            "key":            "FFmpegVideoConvertor",
            "preferedformat": "mp4",
        }],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info       = ydl.extract_info(url, download=True)
            final_path = ydl.prepare_filename(info)
            if not Path(final_path).exists():
                final_path = str(Path(final_path).with_suffix(".mp4"))
            if not Path(final_path).exists() and downloaded_path:
                final_path = downloaded_path[0]

        filename = Path(final_path).name
        update_dl(job_id, status="done", progress=100,
                  message=f"Ready: {filename}",
                  filename=filename, filepath=str(final_path))
    except Exception as e:
        update_dl(job_id, status="error", progress=0, message=_strip_ansi(str(e)))

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    vibe           = request.form.get("vibe", "viral")
    custom_vibe    = request.form.get("custom_vibe", "")
    preset         = request.form.get("preset", "tiktok")
    music_track    = request.form.get("music", "none")
    enable_captions= request.form.get("captions", "true").lower() == "true"
    enable_zoom    = request.form.get("zoom", "true").lower() == "true"
    exclude_json   = request.form.get("exclude", "")
    url            = request.form.get("url", "").strip()

    exclude_ranges = []
    if exclude_json:
        try:
            exclude_ranges = json.loads(exclude_json)
        except Exception:
            pass

    # Resolve video path
    existing_path = request.form.get("video_path", "").strip()
    if url:
        # Inline YouTube URL — download first synchronously (but in a temp job)
        return jsonify({"error": "Use inline_url endpoint for URL input"}), 400
    elif existing_path:
        candidate = Path(existing_path).resolve()
        if not str(candidate).startswith(str(UPLOAD_DIR.resolve())):
            return jsonify({"error": "Invalid video path"}), 400
        if not candidate.exists():
            return jsonify({"error": "File not found"}), 400
        video_path = str(candidate)
    elif "video" in request.files and request.files["video"].filename:
        video_file = request.files["video"]
        ext        = Path(video_file.filename).suffix or ".mp4"
        video_path = str(UPLOAD_DIR / f"{uuid.uuid4()}{ext}")
        video_file.save(video_path)
    else:
        return jsonify({"error": "No video provided"}), 400

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "progress": 0,
                    "message": "Queued…", "clips": [],
                    "video_path": video_path, "conversation": []}

    threading.Thread(
        target=process_video,
        args=(job_id, video_path, vibe, custom_vibe, preset,
              music_track, enable_captions, enable_zoom, exclude_ranges or None),
        daemon=True,
    ).start()

    return jsonify({"job_id": job_id})


@app.route("/process-url", methods=["POST"])
def process_url():
    """Download a URL then immediately start clipping."""
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "No URL"}), 400

    opts = data.get("options", {})
    vibe           = opts.get("vibe", "viral")
    custom_vibe    = opts.get("custom_vibe", "")
    preset         = opts.get("preset", "tiktok")
    music_track    = opts.get("music", "none")
    enable_captions= opts.get("captions", True)
    enable_zoom    = opts.get("zoom", True)

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "downloading", "progress": 2,
                    "message": "Downloading video…", "clips": [],
                    "video_path": "", "conversation": []}

    def worker():
        downloaded: list[str] = []

        def progress_hook(d):
            if d["status"] == "downloading":
                raw = _strip_ansi(d.get("_percent_str", "0%")).replace("%", "").strip()
                try:
                    pct = max(2, min(25, int(float(raw)) // 4))
                except ValueError:
                    pct = jobs[job_id].get("progress", 2)
                speed = _strip_ansi(d.get("_speed_str", "")).strip()
                update_job(job_id, progress=pct,
                           message=f"Downloading… {speed}" if speed else "Downloading…")
            elif d["status"] == "finished":
                update_job(job_id, progress=26, message="Download complete, processing…")
                downloaded.append(d.get("filename", ""))

        outtmpl = str(UPLOAD_DIR / "%(title).80s.%(ext)s")
        ydl_opts = {
            "format":             "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
            "merge_output_format":"mp4",
            "outtmpl":            outtmpl,
            "progress_hooks":     [progress_hook],
            "quiet":              True,
            "no_warnings":        True,
            "noplaylist":         True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info       = ydl.extract_info(url, download=True)
                final_path = ydl.prepare_filename(info)
                if not Path(final_path).exists():
                    final_path = str(Path(final_path).with_suffix(".mp4"))
                if not Path(final_path).exists() and downloaded:
                    final_path = downloaded[0]

            jobs[job_id]["video_path"] = str(final_path)
            process_video(job_id, str(final_path), vibe, custom_vibe, preset,
                          music_track, enable_captions, enable_zoom, None)
        except Exception as e:
            update_job(job_id, status="error", message=_strip_ansi(str(e)))

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/reclip/<job_id>", methods=["POST"])
def reclip(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    video_path = job.get("video_path", "")
    if not video_path or not Path(video_path).exists():
        return jsonify({"error": "Original video no longer available"}), 400

    data           = request.get_json(silent=True) or {}
    vibe           = data.get("vibe", "viral")
    custom_vibe    = data.get("custom_vibe", "")
    preset         = data.get("preset", "tiktok")
    music_track    = data.get("music", "none")
    enable_captions= data.get("captions", True)
    enable_zoom    = data.get("zoom", True)

    # Exclude current clip ranges so Claude picks different moments
    exclude_ranges = [(c["start"], c["end"]) for c in job.get("clips", [])]

    new_job_id = str(uuid.uuid4())
    jobs[new_job_id] = {"status": "queued", "progress": 0,
                        "message": "Queued for re-analysis…", "clips": [],
                        "video_path": video_path, "conversation": []}

    threading.Thread(
        target=process_video,
        args=(new_job_id, video_path, vibe, custom_vibe, preset,
              music_track, enable_captions, enable_zoom, exclude_ranges),
        daemon=True,
    ).start()

    return jsonify({"job_id": new_job_id})


@app.route("/chat/<job_id>", methods=["POST"])
def chat(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    data    = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400

    result = chat_with_claude(job_id, message)

    # Update conversation history
    conversation = job.get("conversation", [])
    conversation.append({"role": "user", "content": message})
    conversation.append({"role": "assistant", "content":
                          json.dumps(result.get("clips")) if result["type"] == "clips"
                          else result["reply"]})
    jobs[job_id]["conversation"] = conversation[-20:]  # keep last 10 exchanges

    # If new clips, export them
    if result["type"] == "clips":
        video_path = job.get("video_path", "")
        preset     = job.get("preset", "tiktok")
        whisper_r  = job.get("_whisper_result")  # may be None
        if video_path and Path(video_path).exists():
            output_dir = CLIPS_DIR / job_id
            output_dir.mkdir(exist_ok=True)
            new_clips = []
            base_idx  = len(job.get("clips", []))
            for i, m in enumerate(result["clips"]):
                out = export_clip(video_path, float(m["start"]), float(m["end"]),
                                  m.get("title", f"reclip_{i}"),
                                  base_idx + i, preset, output_dir,
                                  whisper_result=whisper_r)
                if out:
                    new_clips.append({
                        "file":         Path(out).name,
                        "title":        m.get("title", f"reclip_{i}"),
                        "reason":       m.get("reason", ""),
                        "tags":         m.get("tags", []),
                        "start":        float(m["start"]),
                        "end":          float(m["end"]),
                        "peak_moment":  float(m.get("peak_moment", (m["start"] + m["end"]) / 2)),
                        "download_url": f"/clips/{job_id}/{Path(out).name}",
                        "preview_url":  f"/preview/{job_id}/{Path(out).name}",
                    })
            result["new_clips"] = new_clips

    return jsonify(result)


@app.route("/trim", methods=["POST"])
def trim():
    data     = request.get_json(silent=True) or {}
    job_id   = data.get("job_id", "")
    clip_idx = int(data.get("clip_index", 0))
    new_start= float(data.get("start", 0))
    new_end  = float(data.get("end", 10))

    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    video_path = job.get("video_path", "")
    if not video_path or not Path(video_path).exists():
        return jsonify({"error": "Video not found"}), 400

    preset    = data.get("preset", "tiktok")
    music     = data.get("music", "none")
    captions  = data.get("captions", True)
    zoom      = data.get("zoom", False)
    title     = data.get("title", f"retrim_{clip_idx}")

    output_dir = CLIPS_DIR / job_id
    output_dir.mkdir(exist_ok=True)
    whisper_r  = job.get("_whisper_result")

    out = export_clip(video_path, new_start, new_end, f"trim_{title}",
                      clip_idx + 100, preset, output_dir,
                      whisper_result=whisper_r, music_track=music,
                      enable_zoom=zoom, enable_captions=captions)
    if not out:
        return jsonify({"error": "Export failed"}), 500

    return jsonify({
        "download_url": f"/clips/{job_id}/{Path(out).name}",
        "preview_url":  f"/preview/{job_id}/{Path(out).name}",
        "file": Path(out).name,
    })


@app.route("/status/<job_id>")
def status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Not found"}), 404
    # Don't send large transcript in poll responses
    safe = {k: v for k, v in job.items()
            if k not in ("transcript_text", "_whisper_result", "conversation")}
    safe["has_conversation"] = len(job.get("conversation", [])) > 0
    return jsonify(safe)


@app.route("/clips/<job_id>/<filename>")
def download_clip(job_id: str, filename: str):
    return send_from_directory(str(CLIPS_DIR / job_id), filename, as_attachment=True)


@app.route("/preview/<job_id>/<filename>")
def preview_clip(job_id: str, filename: str):
    return send_from_directory(str(CLIPS_DIR / job_id), filename)


@app.route("/original/<job_id>")
def serve_original(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Not found"}), 404
    vp = job.get("video_path", "")
    if not vp or not Path(vp).exists():
        return jsonify({"error": "Video not found"}), 404
    return send_from_directory(str(Path(vp).parent), Path(vp).name)


@app.route("/uploads")
def list_uploads():
    exts = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    files = []
    for f in sorted(UPLOAD_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if f.suffix.lower() in exts:
            files.append({"name": f.name, "path": str(f),
                          "size_mb": round(f.stat().st_size / 1024 / 1024, 1)})
    return jsonify(files)


@app.route("/download", methods=["POST"])
def start_download():
    data = request.get_json(silent=True) or {}
    url  = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "No URL"}), 400
    job_id = str(uuid.uuid4())
    dl_jobs[job_id] = {"status": "downloading", "progress": 0,
                       "message": "Starting…", "filename": None}
    threading.Thread(target=run_download, args=(job_id, url), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/download-status/<job_id>")
def download_status(job_id: str):
    job = dl_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Not found"}), 404
    return jsonify(job)


@app.route("/history")
def get_history():
    data = load_history()
    return jsonify(data)


@app.route("/music")
def list_music():
    tracks = []
    for f in sorted(MUSIC_DIR.iterdir()):
        if f.suffix == ".mp3":
            tracks.append({"id": f.stem, "label": f.stem.title()})
    return jsonify(tracks)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
