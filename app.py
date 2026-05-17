import subprocess, base64, json, os, uuid, threading, re, shutil
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, render_template

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

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

jobs:    dict[str, dict] = {}
dl_jobs: dict[str, dict] = {}
history_lock = threading.Lock()
_whisper_model = None

CLIP_PADDING   = 2
FRAME_INTERVAL = 2

PRESETS = {
    "tiktok":  {"w": 1080, "h": 1920, "label": "TikTok 9:16"},
    "shorts":  {"w": 1080, "h": 1920, "label": "YouTube Shorts 9:16"},
    "reels":   {"w": 1080, "h": 1920, "label": "Instagram Reels 9:16"},
    "youtube": {"w": 1920, "h": 1080, "label": "YouTube 16:9"},
}

# Vibe → (audio_weight, scene_weight)
VIBE_WEIGHTS = {
    "gaming":    (0.35, 0.65),   # fast cuts matter most
    "funny":     (0.55, 0.45),   # balanced
    "emotional": (0.75, 0.25),   # sustained audio peaks
    "viral":     (0.55, 0.45),   # balanced
    "action":    (0.25, 0.75),   # scene density dominant
}

# ---------------------------------------------------------------------------
# Startup: generate background music with FFmpeg
# ---------------------------------------------------------------------------

def _gen_music():
    tracks = {
        "lofi":     ("aevalsrc=0.15*sin(2*PI*110*t)+0.1*sin(2*PI*165*t)+0.08*sin(2*PI*220*t):s=44100",
                     "volume=0.5,lowpass=f=1800"),
        "hype":     ("aevalsrc=0.18*sin(2*PI*220*t)+0.12*sin(2*PI*330*t)+0.08*sin(2*PI*440*t):s=44100",
                     "volume=0.5"),
        "chill":    ("anoisesrc=color=brown:amplitude=0.04:s=44100", "lowpass=f=500,volume=3"),
        "dramatic": ("aevalsrc=0.15*sin(2*PI*146*t)+0.1*sin(2*PI*175*t)+0.08*sin(2*PI*220*t):s=44100",
                     "volume=0.5,lowpass=f=2500"),
    }
    for name, (src, af) in tracks.items():
        fp = MUSIC_DIR / f"{name}.mp3"
        if not fp.exists():
            subprocess.run(["ffmpeg", "-f", "lavfi", "-i", src, "-t", "120",
                            "-af", af, str(fp), "-y"], capture_output=True)

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
# Whisper (for captions only — no analysis)
# ---------------------------------------------------------------------------

_whisper_model_name = "small"

def get_whisper_model(model_name: str = "small"):
    global _whisper_model, _whisper_model_name
    if _whisper_model is None or _whisper_model_name != model_name:
        if WHISPER_AVAILABLE:
            _whisper_model = openai_whisper.load_model(model_name)
            _whisper_model_name = model_name
    return _whisper_model

def transcribe(video_path: str, model_name: str = "small") -> dict | None:
    model = get_whisper_model(model_name)
    if model is None:
        return None
    try:
        return model.transcribe(video_path, word_timestamps=True, verbose=False)
    except Exception:
        return None

# ---------------------------------------------------------------------------
# LOCAL ANALYSIS — PySceneDetect + MoviePy (no API key needed)
# ---------------------------------------------------------------------------

def _analyze_audio(video_path: str, job_id: str) -> tuple:
    """
    Use MoviePy to extract audio and compute per-frame RMS loudness.
    Returns (times_array, rms_array) both normalised 0-1.
    """
    update_job(job_id, progress=20, message="Analysing audio with MoviePy…")
    try:
        from moviepy import VideoFileClip
        import numpy as np

        clip = VideoFileClip(video_path)
        if clip.audio is None:
            clip.close()
            return np.array([]), np.array([])

        fps = 20           # 20 samples/s → 0.05 s resolution
        arr = clip.audio.to_soundarray(fps=fps)
        clip.close()

        if arr.ndim > 1:
            arr = arr.mean(axis=1)   # stereo → mono

        # RMS in 0.5 s windows (10 samples at 20 Hz)
        chunk = max(1, fps // 2)
        n     = len(arr) // chunk
        rms   = np.array([
            float(np.sqrt(np.mean(arr[i * chunk:(i + 1) * chunk] ** 2)))
            for i in range(n)
        ])
        times = np.arange(n) * (chunk / fps)

        mx = rms.max()
        if mx > 0:
            rms = rms / mx

        return times, rms

    except Exception as e:
        update_job(job_id, progress=20, message=f"Audio analysis skipped ({e})")
        import numpy as np
        return np.array([]), np.array([])


def _detect_scenes(video_path: str, job_id: str) -> list[float]:
    """
    Use PySceneDetect to find scene-cut timestamps (seconds).
    Returns a sorted list of cut times.
    """
    update_job(job_id, progress=35, message="Detecting scene changes with PySceneDetect…")
    try:
        from scenedetect import detect, ContentDetector
        scene_list = detect(video_path, ContentDetector(threshold=27.0),
                            show_progress=False)
        # scene_list[0] always starts at 0; cuts are the start of scenes 1+
        cuts = [scene[0].get_seconds() for scene in scene_list[1:]]
        return cuts
    except Exception as e:
        update_job(job_id, progress=35, message=f"Scene detection skipped ({e})")
        return []


def _auto_tags(audio_mean: float, cuts_in_clip: int,
               duration: float, score: float) -> list[str]:
    cps  = cuts_in_clip / duration if duration > 0 else 0
    tags: list[str] = []

    if score > 0.70:            tags.append("Hype")
    if audio_mean > 0.65 and cps < 0.25:  tags.append("Emotional")
    if cps > 0.45:              tags.append("Surprising")
    if audio_mean > 0.50 and cps > 0.25:  tags.append("Funny")
    if audio_mean < 0.30 and cps > 0.35:  tags.append("Dramatic")
    if 0.35 < audio_mean < 0.60 and cps < 0.15: tags.append("Quotable")
    if audio_mean > 0.80:       tags.append("Wholesome")

    seen, result = set(), []
    for t in tags:
        if t not in seen:
            seen.add(t); result.append(t)
        if len(result) == 3:
            break
    return result or ["Surprising"]


def _auto_reason(audio_mean: float, cuts_in_clip: int,
                 duration: float, score: float) -> str:
    parts = []
    pct = int(audio_mean * 100)
    if pct > 70:
        parts.append(f"loud audio ({pct}% peak)")
    elif pct > 40:
        parts.append(f"elevated audio ({pct}%)")
    if cuts_in_clip > 5:
        parts.append(f"{cuts_in_clip} rapid scene cuts")
    elif cuts_in_clip > 1:
        parts.append(f"{cuts_in_clip} scene transitions")
    if score > 0.80:
        parts.append("very high combined energy")
    elif score > 0.50:
        parts.append("high combined energy")
    return ("Detected: " + ", ".join(parts)) if parts else f"Combined viral score {score:.0%}"


def find_moments_local(video_path: str, vibe: str, job_id: str,
                       alt_pass: bool = False) -> list[dict]:
    """
    Pure-local clip finder using PySceneDetect + MoviePy.
    No internet or API key required.
    """
    import numpy as np

    duration = get_video_duration(video_path)
    if duration < 5:
        return []

    RES  = 0.5   # seconds per energy bin
    BINS = int(duration / RES) + 1

    # ── 1. Audio signal ──────────────────────────────────────────────────
    audio_times, audio_rms = _analyze_audio(video_path, job_id)
    audio_sig = np.zeros(BINS)
    for t, v in zip(audio_times, audio_rms):
        idx = int(t / RES)
        if 0 <= idx < BINS:
            audio_sig[idx] = v

    # ── 2. Scene-density signal ──────────────────────────────────────────
    cuts    = _detect_scenes(video_path, job_id)
    scene_sig = np.zeros(BINS)
    sigma   = max(1, int(1.0 / RES))  # 1-second gaussian — keeps peaks sharp
    for ct in cuts:
        ci = int(ct / RES)
        for off in range(-sigma * 3, sigma * 3 + 1):
            j = ci + off
            if 0 <= j < BINS:
                scene_sig[j] += float(np.exp(-0.5 * (off / sigma) ** 2))
    if scene_sig.max() > 0:
        scene_sig /= scene_sig.max()

    # ── 3. Combined energy ───────────────────────────────────────────────
    wa, ws = VIBE_WEIGHTS.get(vibe, (0.55, 0.45))
    if alt_pass:
        wa, ws = ws, wa

    energy = wa * audio_sig + ws * scene_sig
    # Light smoothing: 1 s window preserves local structure
    k_bins = max(1, int(1.0 / RES))
    kernel = np.ones(k_bins) / k_bins
    energy = np.convolve(energy, kernel, mode="same")

    # Subtract a slow-moving baseline so relative peaks stand out over
    # sustained high-energy sections (common in gaming videos with BGM)
    baseline_bins = max(1, int(30.0 / RES))
    baseline_kernel = np.ones(baseline_bins) / baseline_bins
    baseline = np.convolve(energy, baseline_kernel, mode="same")
    energy = np.clip(energy - 0.5 * baseline, 0, None)

    update_job(job_id, progress=55, message="Scoring and selecting best moments…")

    # ── 4. Peak detection (greedy, adaptive gap) ─────────────────────────
    # Gap scales with video length so we always find several clips
    min_gap_s = max(15.0, min(45.0, duration / 12))
    min_gap   = int(min_gap_s / RES)

    def _pick_peaks(gap: int, limit: int) -> list[tuple[int, float]]:
        raw: list[tuple[int, float]] = []
        for i in range(1, BINS - 1):
            if energy[i] > energy[i - 1] and energy[i] >= energy[i + 1]:
                raw.append((i, float(energy[i])))
        raw.sort(key=lambda x: x[1], reverse=True)
        out, taken = [], []
        for idx, score in raw:
            if not any(abs(idx - t) < gap for t in taken):
                out.append((idx, score))
                taken.append(idx)
            if len(out) >= limit:
                break
        return out

    selected = _pick_peaks(min_gap, 6)
    # Fallback: halve the gap to get at least 3 clips
    if len(selected) < 3 and duration > 30:
        selected = _pick_peaks(max(1, min_gap // 2), 6)

    # ── 5. Build clip windows ────────────────────────────────────────────
    moments: list[dict] = []
    for peak_idx, peak_score in selected:
        peak_t = peak_idx * RES

        raw_s = max(0.0,      peak_t - 8.0)
        raw_e = min(duration, peak_t + 10.0)

        # Snap start back to a nearby scene cut (within 5 s)
        snap_s = raw_s
        for ct in sorted(cuts):
            if raw_s - 5 <= ct <= raw_s:
                snap_s = ct
        # Snap end forward to a nearby scene cut (within 5 s)
        snap_e = raw_e
        for ct in sorted(cuts, reverse=True):
            if raw_e <= ct <= raw_e + 5:
                snap_e = ct
                break

        # Enforce duration bounds
        cdur = snap_e - snap_s
        if cdur < 4:
            snap_e = min(duration, snap_s + 10.0)
        elif cdur > 40:
            snap_e = snap_s + 30.0

        # Profile for tagging
        si = int(snap_s / RES)
        ei = min(BINS, int(snap_e / RES))
        clip_audio = float(np.mean(audio_sig[si:ei])) if ei > si else 0.0
        cuts_in    = sum(1 for ct in cuts if snap_s <= ct <= snap_e)
        clip_dur   = snap_e - snap_s

        moments.append({
            "start":       round(snap_s, 1),
            "end":         round(snap_e, 1),
            "peak_moment": round(peak_t, 1),
            "title":       f"moment_{int(peak_t)}s",
            "reason":      _auto_reason(clip_audio, cuts_in, clip_dur, peak_score),
            "tags":        _auto_tags(clip_audio, cuts_in, clip_dur, peak_score),
            "viral_score": round(peak_score * 100, 1),
        })

    moments.sort(key=lambda m: m["start"])
    return moments


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_video_duration(video_path: str) -> float:
    r = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", video_path,
    ], capture_output=True, text=True)
    try:
        return float(json.loads(r.stdout)["format"]["duration"])
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise ValueError(f"Could not read duration from {video_path}: {e}") from e


def update_job(job_id: str, **kwargs):
    if job_id in jobs:
        jobs[job_id].update(kwargs)


def update_dl(job_id: str, **kwargs):
    if job_id in dl_jobs:
        dl_jobs[job_id].update(kwargs)


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)

# ---------------------------------------------------------------------------
# Caption generation (ASS)
# ---------------------------------------------------------------------------

def _ass_time(s: float) -> str:
    h  = int(s // 3600)
    m  = int((s % 3600) // 60)
    sc = s % 60
    return f"{h}:{m:02d}:{sc:05.2f}"


def generate_ass(whisper_result: dict, clip_start: float, clip_end: float,
                 path: str, vertical: bool = True, caption_style: str = "karaoke"):
    w, h  = (1080, 1920) if vertical else (1920, 1080)
    fs    = 78 if vertical else 56
    mv    = 280 if vertical else 90
    dur   = clip_end - clip_start

    # Karaoke: white pre-highlight → yellow on spoken word
    # Classic: plain white
    if caption_style == "karaoke":
        primary   = "&H0000FFFF"   # yellow (BGR: 00FFFF → R=FF G=FF B=00)
        secondary = "&H00FFFFFF"   # white  (pre-highlight)
    else:
        primary   = "&H00FFFFFF"
        secondary = "&H000000FF"

    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {w}\nPlayResY: {h}\nWrapStyle: 1\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, Strikeout, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,Arial Black,{fs},{primary},{secondary},&H00000000,"
        f"&H80000000,-1,0,0,0,100,100,0,0,1,5,2,2,40,40,{mv},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    events: list[str] = []
    segs   = whisper_result.get("segments", [])

    all_words: list[dict] = []
    for seg in segs:
        for w_obj in seg.get("words", []):
            ws   = w_obj.get("start", 0)
            we   = w_obj.get("end", 0)
            word = w_obj.get("word", "").strip()
            if word and we > clip_start and ws < clip_end:
                all_words.append({"word": word,
                                   "start": ws - clip_start,
                                   "end":   we - clip_start})

    def _esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")

    if all_words:
        for i in range(0, len(all_words), 4):
            group = all_words[i:i + 4]
            t0 = max(0.0, group[0]["start"])
            t1 = min(dur,  group[-1]["end"])
            if t0 >= dur:
                continue

            if caption_style == "karaoke":
                # {\kN} = N centiseconds for this word; libass sweeps
                # secondary→primary colour word-by-word
                parts = []
                cursor = t0
                for w_obj in group:
                    gap_cs  = max(0, int((w_obj["start"] - cursor) * 100))
                    word_cs = max(10, int((w_obj["end"] - w_obj["start"]) * 100))
                    if gap_cs > 0 and parts:
                        # absorb leading gap into previous word's tag duration
                        parts[-1] = parts[-1]   # no-op; gap already falls inside prev end
                    parts.append(f"{{\\k{word_cs}}}{_esc(w_obj['word'].upper())}")
                    cursor = w_obj["end"]
                text = " ".join(parts)
            else:
                text = _esc(" ".join(w["word"].upper() for w in group))

            events.append(
                f"Dialogue: 0,{_ass_time(t0)},{_ass_time(t1)},Default,,0,0,0,,{text}"
            )
    else:
        for seg in segs:
            ss, se = seg["start"], seg["end"]
            if se <= clip_start or ss >= clip_end:
                continue
            t0   = max(0.0, ss - clip_start)
            t1   = min(dur,  se - clip_start)
            events.append(
                f"Dialogue: 0,{_ass_time(t0)},{_ass_time(t1)},Default,,0,0,0,,{_esc(seg['text'].strip().upper())}"
            )

    Path(path).write_text(header + "\n".join(events), encoding="utf-8")

# ---------------------------------------------------------------------------
# Clip export (two-pass: format + zoom → captions + music)
# ---------------------------------------------------------------------------

def export_clip(video_path: str, start: float, end: float, title: str, index: int,
                preset: str, output_dir: Path,
                whisper_result: dict | None = None,
                music_track: str | None = None,
                enable_zoom: bool = True,
                enable_captions: bool = True,
                caption_style: str = "karaoke") -> str:

    safe = re.sub(r"[^a-zA-Z0-9_]", "_", title)[:40]
    cfg  = PRESETS.get(preset, PRESETS["tiktok"])
    w, h = cfg["w"], cfg["h"]
    vert = (h > w)

    ps  = max(0.0, start - CLIP_PADDING)
    pe  = end + CLIP_PADDING
    dur = pe - ps
    pk  = (start - ps) + (end - start) / 2.0

    temp  = str(output_dir / f"_tmp_{index}.mp4")
    final = str(output_dir / f"clip_{index:02d}_{safe}.mp4")

    # ── Pass 1: cut + format + optional zoom ─────────────────────────────
    crop_vf = (f"crop=ih*9/16:ih,scale={w}:{h}" if vert
               else f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                    f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2")

    if enable_zoom:
        fps = 30
        zi  = max(0, int((pk - 0.5) * fps))
        zo  = int((pk + 0.6) * fps)
        ze  = (f"if(between(on,{zi},{zo}),"
               f"min(zoom+0.005,1.1),"
               f"max(zoom-0.005,1))")
        vf1 = (f"{crop_vf},fps={fps},"
               f"zoompan=z='{ze}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
               f":d=1:s={w}x{h}:fps={fps}")
    else:
        vf1 = crop_vf

    subprocess.run([
        "ffmpeg", "-ss", str(ps), "-t", str(dur), "-i", video_path,
        "-vf", vf1, "-c:v", "libx264", "-crf", "20", "-r", "30",
        "-c:a", "aac", "-b:a", "192k", temp, "-y",
    ], capture_output=True)

    if not Path(temp).exists():
        return ""

    # ── Pass 2: captions + music ─────────────────────────────────────────
    cap_file = None
    if enable_captions and whisper_result:
        cap_file = str(output_dir / f"_cap_{index}.ass")
        generate_ass(whisper_result, ps, pe, cap_file, vert, caption_style)

    music_path = None
    if music_track and music_track != "none":
        mp = MUSIC_DIR / f"{music_track}.mp3"
        if mp.exists():
            music_path = str(mp)

    has_cap   = cap_file and Path(cap_file).exists()
    has_music = bool(music_path)

    if not has_cap and not has_music:
        Path(temp).rename(final)
        return final

    inputs = ["ffmpeg", "-i", temp]
    if has_music:
        inputs += ["-i", music_path]

    if has_cap and has_music:
        fc = (f"[0:v]subtitles={cap_file}[vout];"
              f"[1:a]atrim=0:{dur:.2f},asetpts=PTS-STARTPTS,volume=0.18[mus];"
              f"[0:a][mus]amix=inputs=2:duration=first[aout]")
        cmd2 = inputs + ["-filter_complex", fc,
                         "-map", "[vout]", "-map", "[aout]",
                         "-c:v", "libx264", "-crf", "20", "-c:a", "aac",
                         final, "-y"]
    elif has_cap:
        cmd2 = inputs + ["-vf", f"subtitles={cap_file}",
                         "-c:v", "libx264", "-crf", "20", "-c:a", "copy",
                         final, "-y"]
    else:
        fc   = (f"[1:a]atrim=0:{dur:.2f},asetpts=PTS-STARTPTS,volume=0.18[mus];"
                f"[0:a][mus]amix=inputs=2:duration=first[aout]")
        cmd2 = inputs + ["-filter_complex", fc,
                         "-map", "0:v", "-map", "[aout]",
                         "-c:v", "copy", "-c:a", "aac", final, "-y"]

    result = subprocess.run(cmd2, capture_output=True)

    # Fallback: retry without captions if subtitles filter fails
    if result.returncode != 0 and has_cap:
        if has_music:
            fc   = (f"[1:a]atrim=0:{dur:.2f},asetpts=PTS-STARTPTS,volume=0.18[mus];"
                    f"[0:a][mus]amix=inputs=2:duration=first[aout]")
            cmd3 = ["ffmpeg", "-i", temp, "-i", music_path,
                    "-filter_complex", fc, "-map", "0:v", "-map", "[aout]",
                    "-c:v", "copy", "-c:a", "aac", final, "-y"]
        else:
            cmd3 = ["ffmpeg", "-i", temp, "-c", "copy", final, "-y"]
        subprocess.run(cmd3, capture_output=True)

    Path(temp).unlink(missing_ok=True)
    if cap_file:
        Path(cap_file).unlink(missing_ok=True)

    return final if Path(final).exists() else ""

# ---------------------------------------------------------------------------
# Main processing worker
# ---------------------------------------------------------------------------

def process_video(job_id: str, video_path: str, vibe: str,
                  preset: str, music_track: str,
                  enable_captions: bool, enable_zoom: bool,
                  alt_pass: bool = False,
                  caption_style: str = "karaoke",
                  whisper_model: str = "small"):
    try:
        # 1. Transcribe (Whisper — free/local, used only for captions)
        whisper_result = None
        if WHISPER_AVAILABLE and enable_captions:
            update_job(job_id, status="transcribing", progress=10,
                       message=f"Transcribing audio with Whisper ({whisper_model})…")
            whisper_result = transcribe(video_path, whisper_model)

        # 2. Local analysis — PySceneDetect + MoviePy
        update_job(job_id, status="analyzing", progress=15,
                   message="Starting local video analysis…")
        moments = find_moments_local(video_path, vibe, job_id, alt_pass=alt_pass)

        if not moments:
            update_job(job_id, status="error", progress=0,
                       message="No moments detected. Try a different vibe or a longer video.")
            return

        # 3. Export clips
        update_job(job_id, status="exporting", progress=60,
                   message=f"Found {len(moments)} moments. Exporting clips…")

        out_dir = CLIPS_DIR / job_id
        out_dir.mkdir(exist_ok=True)

        clips = []
        for i, m in enumerate(moments):
            pct = 60 + int((i / len(moments)) * 35)
            update_job(job_id, progress=pct,
                       message=f"Exporting clip {i + 1}/{len(moments)} "
                               f"({m['title']}, viral score {m['viral_score']})…")
            out = export_clip(
                video_path,
                float(m["start"]), float(m["end"]),
                m["title"], i, preset, out_dir,
                whisper_result=whisper_result,
                music_track=music_track,
                enable_zoom=enable_zoom,
                enable_captions=enable_captions,
                caption_style=caption_style,
            )
            if not out:
                continue
            clips.append({
                "file":         Path(out).name,
                "title":        m["title"],
                "reason":       m["reason"],
                "tags":         m["tags"],
                "viral_score":  m["viral_score"],
                "start":        float(m["start"]),
                "end":          float(m["end"]),
                "peak_moment":  float(m["peak_moment"]),
                "download_url": f"/clips/{job_id}/{Path(out).name}",
                "preview_url":  f"/preview/{job_id}/{Path(out).name}",
            })

        jobs[job_id].update({
            "status":     "done",
            "progress":   100,
            "message":    "All clips exported!",
            "clips":      clips,
            "video_path": video_path,
        })

        append_history({
            "id":         job_id,
            "created_at": datetime.utcnow().isoformat(),
            "video_name": Path(video_path).name,
            "vibe":       vibe,
            "preset":     preset,
            "clip_count": len(clips),
            "clips":      clips,
        })

    except Exception as e:
        shutil.rmtree(FRAMES_DIR / job_id, ignore_errors=True)
        update_job(job_id, status="error", progress=0, message=str(e))

# ---------------------------------------------------------------------------
# yt-dlp download worker
# ---------------------------------------------------------------------------

def run_download(job_id: str, url: str):
    downloaded: list[str] = []

    def progress_hook(d):
        if d["status"] == "downloading":
            raw  = _strip_ansi(d.get("_percent_str", "0%")).replace("%", "").strip()
            try:   pct = min(95, int(float(raw)))
            except ValueError: pct = dl_jobs[job_id].get("progress", 0)
            speed = _strip_ansi(d.get("_speed_str", "")).strip()
            eta   = _strip_ansi(d.get("_eta_str", "")).strip()
            msg   = f"Downloading… {pct}%"
            if speed: msg += f"  {speed}"
            if eta and eta not in ("Unknown", "--:--"): msg += f"  ETA {eta}"
            update_dl(job_id, progress=pct, message=msg)
        elif d["status"] == "finished":
            update_dl(job_id, progress=96, message="Finalising…")
            downloaded.append(d.get("filename", ""))

    outtmpl  = str(UPLOAD_DIR / "%(title).80s.%(ext)s")
    ydl_opts = {
        "format":              "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]"
                               "/bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
        "merge_output_format": "mp4",
        "outtmpl":             outtmpl,
        "progress_hooks":      [progress_hook],
        "quiet":               True,
        "no_warnings":         True,
        "noplaylist":          True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info       = ydl.extract_info(url, download=True)
            final_path = ydl.prepare_filename(info)
            if not Path(final_path).exists():
                final_path = str(Path(final_path).with_suffix(".mp4"))
            if not Path(final_path).exists() and downloaded:
                final_path = downloaded[0]

        filename = Path(final_path).name
        update_dl(job_id, status="done", progress=100,
                  message=f"Ready: {filename}",
                  filename=filename, filepath=str(final_path))
    except Exception as e:
        update_dl(job_id, status="error", progress=0,
                  message=_strip_ansi(str(e)))

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    vibe            = request.form.get("vibe", "viral")
    preset          = request.form.get("preset", "tiktok")
    music_track     = request.form.get("music", "none")
    enable_captions = request.form.get("captions", "true").lower() == "true"
    enable_zoom     = request.form.get("zoom", "true").lower() == "true"
    caption_style   = request.form.get("caption_style", "karaoke")
    whisper_model   = request.form.get("whisper_model", "small")

    existing_path = request.form.get("video_path", "").strip()
    if existing_path:
        candidate = Path(existing_path).resolve()
        if not str(candidate).startswith(str(UPLOAD_DIR.resolve())):
            return jsonify({"error": "Invalid video path"}), 400
        if not candidate.exists():
            return jsonify({"error": "File not found"}), 400
        video_path = str(candidate)
    elif "video" in request.files and request.files["video"].filename:
        f   = request.files["video"]
        ext = Path(f.filename).suffix or ".mp4"
        video_path = str(UPLOAD_DIR / f"{uuid.uuid4()}{ext}")
        f.save(video_path)
    else:
        return jsonify({"error": "No video provided"}), 400

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "progress": 0,
                    "message": "Queued…", "clips": [],
                    "video_path": video_path}

    threading.Thread(
        target=process_video,
        args=(job_id, video_path, vibe, preset, music_track,
              enable_captions, enable_zoom, False,
              caption_style, whisper_model),
        daemon=True,
    ).start()

    return jsonify({"job_id": job_id})


@app.route("/process-url", methods=["POST"])
def process_url():
    data = request.get_json(silent=True) or {}
    url  = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "No URL"}), 400

    opts = data.get("options", {})
    vibe            = opts.get("vibe", "viral")
    preset          = opts.get("preset", "tiktok")
    music_track     = opts.get("music", "none")
    enable_captions = opts.get("captions", True)
    enable_zoom     = opts.get("zoom", True)

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "downloading", "progress": 2,
                    "message": "Downloading video…", "clips": [],
                    "video_path": ""}

    def worker():
        downloaded: list[str] = []

        def ph(d):
            if d["status"] == "downloading":
                raw  = _strip_ansi(d.get("_percent_str", "0%")).replace("%", "").strip()
                try:   pct = max(2, min(25, int(float(raw)) // 4))
                except ValueError: pct = 2
                speed = _strip_ansi(d.get("_speed_str", "")).strip()
                update_job(job_id, progress=pct,
                           message=f"Downloading… {speed}" if speed else "Downloading…")
            elif d["status"] == "finished":
                update_job(job_id, progress=26, message="Download complete, analysing…")
                downloaded.append(d.get("filename", ""))

        outtmpl  = str(UPLOAD_DIR / "%(title).80s.%(ext)s")
        ydl_opts = {
            "format":              "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]"
                                   "/bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
            "merge_output_format": "mp4",
            "outtmpl":             outtmpl,
            "progress_hooks":      [ph],
            "quiet":               True,
            "no_warnings":         True,
            "noplaylist":          True,
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
            process_video(job_id, str(final_path), vibe, preset,
                          music_track, enable_captions, enable_zoom)
        except Exception as e:
            update_job(job_id, status="error",
                       message=_strip_ansi(str(e)))

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

    data            = request.get_json(silent=True) or {}
    vibe            = data.get("vibe", "viral")
    preset          = data.get("preset", "tiktok")
    music_track     = data.get("music", "none")
    enable_captions = data.get("captions", True)
    enable_zoom     = data.get("zoom", True)

    new_id = str(uuid.uuid4())
    jobs[new_id] = {"status": "queued", "progress": 0,
                    "message": "Re-analysing with alternate weights…",
                    "clips": [], "video_path": video_path}

    threading.Thread(
        target=process_video,
        args=(new_id, video_path, vibe, preset, music_track,
              enable_captions, enable_zoom, True),   # alt_pass=True
        daemon=True,
    ).start()

    return jsonify({"job_id": new_id})


@app.route("/trim", methods=["POST"])
def trim():
    data      = request.get_json(silent=True) or {}
    job_id    = data.get("job_id", "")
    try:
        clip_idx  = int(data.get("clip_index", 0))
        new_start = float(data.get("start", 0))
        new_end   = float(data.get("end", 10))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid clip_index, start, or end"}), 400

    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    video_path = job.get("video_path", "")
    if not video_path or not Path(video_path).exists():
        return jsonify({"error": "Video not found"}), 400

    preset  = data.get("preset", "tiktok")
    music   = data.get("music", "none")
    captions = data.get("captions", True)
    zoom    = data.get("zoom", False)
    title   = data.get("title", f"retrim_{clip_idx}")

    out_dir = CLIPS_DIR / job_id
    out_dir.mkdir(exist_ok=True)

    out = export_clip(video_path, new_start, new_end, f"trim_{title}",
                      clip_idx + 100, preset, out_dir,
                      music_track=music, enable_zoom=zoom,
                      enable_captions=captions)
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
    return jsonify({k: v for k, v in job.items() if k != "_whisper"})


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
    exts  = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    files = []
    upload_root = UPLOAD_DIR.resolve()
    for f in sorted(UPLOAD_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if f.suffix.lower() not in exts:
            continue
        if not f.resolve().is_relative_to(upload_root):
            continue  # skip symlinks pointing outside uploads/
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
    return jsonify(load_history())


@app.route("/music")
def list_music():
    tracks = [{"id": f.stem, "label": f.stem.title()}
              for f in sorted(MUSIC_DIR.iterdir()) if f.suffix == ".mp3"]
    return jsonify(tracks)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
