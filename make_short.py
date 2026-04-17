#!/usr/bin/env python3
"""
make_short.py — Professional ranked YouTube Short generator.

Mode 1 (auto): python make_short.py <URL|file>
               Auto-detects 5 key moments via motion analysis and cuts
               each clip to its natural action window (3–8 s).

Mode 2 (manual): python make_short.py
                 Reads up to 5 entries from clips.txt; finds the best
                 natural-window moment in each.

Output structure:
  [0.8s intro "TOP 5"] →
  clip #5  → [0.5s "#4"] → clip #4 → [0.5s "#3"] → clip #3 →
  [0.5s "#2"] → clip #2 → [0.5s "#1"] → clip #1 → [0.5s outro]

Requires: ffmpeg (system), yt-dlp (pip), opencv-python (pip), numpy (pip)
"""

import json
import os
import sys
import argparse
import tempfile
import subprocess
import wave

import cv2
import numpy as np


# ── Constants ──────────────────────────────────────────────────────────────────

TARGET_W     = 1080
TARGET_H     = 1920
INTRO_DUR    = 0.8
TRANS_DUR    = 0.5    # black transition screen between clips
OUTRO_DUR    = 0.5
FADE_DUR     = 0.25   # fade to black at end of each clip
MIN_CLIP     = 3.0
MAX_CLIP     = 8.0
NUM_CLIPS    = 5
WHOOSH_DUR   = 0.45
MUSIC_VOL    = 0.10

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "C:/Windows/Fonts/arialbd.ttf",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _has_audio(path: str) -> bool:
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", "-select_streams", "a", path],
        capture_output=True, text=True,
    )
    try:
        return bool(json.loads(r.stdout).get("streams"))
    except Exception:
        return False


def _find_font() -> str | None:
    for fp in FONT_CANDIDATES:
        if os.path.exists(fp):
            return fp
    return None


def _fo() -> str:
    fp = _find_font()
    return f":fontfile='{fp.replace(chr(92), '/').replace(':', chr(92) + ':')}'" if fp else ""


def _run(cmd: list, *, silent: bool = True) -> None:
    subprocess.run(
        cmd,
        check=True,
        stderr=subprocess.DEVNULL if silent else None,
    )


# ── Download ───────────────────────────────────────────────────────────────────

def download_video(url: str, dest: str) -> None:
    print(f"  Downloading: {url}")
    _run([
        sys.executable, "-m", "yt_dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", dest, url,
    ], silent=False)
    print("  Done.")


# ── Motion analysis ────────────────────────────────────────────────────────────

def analyse_video(video_path: str) -> tuple[list, list, float, float]:
    """
    Sample motion scores at ~4 fps using frame-diff.
    Returns (timestamps, scores, fps, duration).
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps
    interval = max(1, int(fps / 4))

    timestamps, scores, prev_gray, idx = [], [], None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % interval == 0:
            sm = cv2.resize(frame, (320, 240))
            gray = cv2.GaussianBlur(cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY), (21, 21), 0)
            if prev_gray is not None:
                timestamps.append(idx / fps)
                scores.append(float(np.mean(cv2.absdiff(prev_gray, gray))))
            prev_gray = gray
        idx += 1

    cap.release()
    return timestamps, scores, fps, duration


def find_top_moments(timestamps: list, scores: list, duration: float,
                     n: int = NUM_CLIPS) -> list:
    """Return n start-timestamps of highest-motion moments, min-spaced."""
    if not scores:
        step = duration / (n + 1)
        return [step * (i + 1) for i in range(n)]

    vals = np.array(scores)
    window = min(16, max(1, len(vals) // 10))
    if window > 1:
        vals = np.convolve(vals, np.ones(window) / window, mode="same")

    ts = np.array(timestamps)
    min_gap = MAX_CLIP + 1.0
    selected = []

    for idx in np.argsort(vals)[::-1]:
        t = float(ts[idx])
        t = max(0.0, min(duration - MIN_CLIP, t))
        if not any(abs(t - s) < min_gap for s in selected):
            selected.append(t)
        if len(selected) == n:
            break

    while len(selected) < n:
        step = duration / (n + 1)
        selected.append(step * (len(selected) + 1))

    selected.sort()
    return selected[:n]


def natural_clip_bounds(timestamps: list, scores: list,
                        center: float, duration: float) -> tuple[float, float]:
    """
    Expand outward from center until motion drops below a local threshold,
    returning (start, end) clamped to [MIN_CLIP, MAX_CLIP] seconds.
    """
    if not timestamps:
        return max(0.0, center), min(duration, center + 5.0)

    ts = np.array(timestamps)
    sc = np.array(scores)
    ci = int(np.argmin(np.abs(ts - center)))

    # Local mean in ±5-second window as the baseline
    mask = (ts >= center - 5.0) & (ts <= center + 5.0)
    base = sc[mask].mean() if mask.sum() > 3 else sc.mean()
    thresh = base * 0.38

    # Walk left to find natural start
    start_idx = ci
    for i in range(ci - 1, max(0, ci - int(MAX_CLIP * 4)), -1):
        if sc[i] < thresh:
            start_idx = i + 1
            break
        start_idx = i

    # Walk right to find natural end
    end_idx = ci
    for i in range(ci + 1, min(len(ts), ci + int(MAX_CLIP * 4))):
        if sc[i] < thresh:
            end_idx = i - 1
            break
        end_idx = i

    s, e = float(ts[start_idx]), float(ts[end_idx])
    dur = e - s

    # Enforce min / max
    if dur < MIN_CLIP:
        mid = (s + e) / 2.0
        s, e = mid - MIN_CLIP / 2, mid + MIN_CLIP / 2
    elif dur > MAX_CLIP:
        mid = (s + e) / 2.0
        s, e = mid - MAX_CLIP / 2, mid + MAX_CLIP / 2

    # Clamp to video bounds
    s = max(0.0, s)
    e = min(duration, e)
    if e - s < MIN_CLIP:
        e = min(duration, s + MIN_CLIP)
        if e == duration:
            s = max(0.0, e - MIN_CLIP)

    return s, e


# ── Black-screen segment generators ───────────────────────────────────────────

def _black_seg(duration: float, vf: str, out: str) -> None:
    """Render a silent black screen with the given video filter applied."""
    _run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=black:size={TARGET_W}x{TARGET_H}:rate=30",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-t", str(duration),
        "-vf", vf,
        "-filter_complex", "[1:a]aresample=44100[a]",
        "-map", "0:v", "-map", "[a]",
        "-c:v", "libx264", "-crf", "20", "-preset", "fast",
        "-c:a", "aac", "-ar", "44100", "-ac", "2",
        out,
    ])


def make_intro(out: str) -> None:
    """0.8 s 'TOP 5' with a subtle zoom-in via zoompan."""
    fo = _fo()
    # Draw text first, then zoom in (z goes 1.0 → 1.3 over 24 frames)
    vf = (
        f"drawtext=text='TOP 5'{fo}:fontsize=260"
        f":fontcolor=white:borderw=18:bordercolor=black"
        f":x=(w-text_w)/2:y=(h-text_h)/2,"
        f"zoompan=z='1.0+0.0125*on':d=1"
        f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        f":s={TARGET_W}x{TARGET_H}"
    )
    _black_seg(INTRO_DUR, vf, out)


def make_transition(rank: int, out: str) -> None:
    """0.5 s black screen announcing the upcoming rank — clean top-right text."""
    fo = _fo()
    vf = (
        f"drawtext=text='#{rank}'{fo}:fontsize=60"
        f":fontcolor=white:borderw=3:bordercolor=black"
        f":x=w-text_w-55:y=55"
    )
    _black_seg(TRANS_DUR, vf, out)


def make_outro(out: str) -> None:
    """0.5 s plain black outro."""
    _black_seg(OUTRO_DUR, "null", out)


# ── Main clip processor ────────────────────────────────────────────────────────

def process_clip(source: str, t_start: float, t_end: float,
                 rank: int, out: str) -> float:
    """
    Extract [t_start, t_end], crop to 9:16, burn a small pill badge (#N),
    fade to black at the end.  Returns actual clip duration written.
    """
    dur = round(t_end - t_start, 3)
    fade_start = max(0.0, dur - FADE_DUR)
    fo = _fo()

    scale_crop = (
        f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=increase,"
        f"crop={TARGET_W}:{TARGET_H}"
    )

    # Semi-transparent dark pill badge, top-right
    badge_bg = (
        f"drawbox=x=w-118:y=20:w=104:h=54"
        f":color=black@0.55:t=fill"
    )
    badge_txt = (
        f"drawtext=text='#{rank}'{fo}:fontsize=38"
        f":fontcolor=white@0.92:borderw=2:bordercolor=black@0.4"
        f":x=w-108:y=29"
    )
    fade_v = f"fade=t=out:st={fade_start}:d={FADE_DUR}"

    vf = f"{scale_crop},{badge_bg},{badge_txt},{fade_v}"
    af = f"afade=t=out:st={fade_start}:d={FADE_DUR}"

    has_audio = _has_audio(source)

    if has_audio:
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(t_start), "-i", source,
            "-t", str(dur),
            "-vf", vf, "-af", af,
            "-map", "0:v", "-map", "0:a",
            "-c:v", "libx264", "-crf", "20", "-preset", "fast",
            "-c:a", "aac", "-ar", "44100", "-ac", "2",
            out,
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(t_start), "-i", source,
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
            "-t", str(dur),
            "-vf", vf,
            "-filter_complex",
            f"[1:a]aresample=44100,afade=t=out:st={fade_start}:d={FADE_DUR}[a]",
            "-map", "0:v", "-map", "[a]",
            "-c:v", "libx264", "-crf", "20", "-preset", "fast",
            "-c:a", "aac", "-ar", "44100", "-ac", "2",
            out,
        ]

    _run(cmd)
    return dur


# ── Whoosh ─────────────────────────────────────────────────────────────────────

def make_whoosh(out: str, sr: int = 44100) -> None:
    n = int(sr * WHOOSH_DUR)
    t = np.linspace(0, WHOOSH_DUR, n, dtype=np.float32)
    freqs = 1400.0 * np.exp(-5.0 * t / WHOOSH_DUR) + 80.0
    tone = np.sin(2 * np.pi * np.cumsum(freqs) / sr)
    noise = np.convolve(
        np.random.normal(0, 1.0, n).astype(np.float32),
        np.ones(30, dtype=np.float32) / 30, mode="same",
    )
    sig = tone * 0.55 + noise * 0.45
    env = np.exp(-5.0 * t / WHOOSH_DUR)
    env[: int(0.03 * sr)] *= np.linspace(0, 1, int(0.03 * sr), dtype=np.float32)
    pcm = np.clip(sig * env * 0.8 * 32767, -32768, 32767).astype(np.int16)
    with wave.open(out, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


# ── Final assembly ─────────────────────────────────────────────────────────────

def assemble(
    intro_path: str,
    segments: list,        # [(trans_path|None, clip_path, clip_dur), ...]
    outro_path: str,
    whoosh_path: str,
    music_path: str | None,
    output_path: str,
    tmp: str,
) -> None:
    """
    Concatenate all segments, mix whoosh at each clip→transition boundary,
    and optionally mix in background music.
    """
    # ── Build concat list ──────────────────────────────────────────────────────
    all_files: list[tuple[str, float]] = [(intro_path, INTRO_DUR)]
    for trans_path, clip_path, clip_dur in segments:
        all_files.append((clip_path, clip_dur))
        if trans_path:
            all_files.append((trans_path, TRANS_DUR))
    all_files.append((outro_path, OUTRO_DUR))

    total_dur = sum(d for _, d in all_files)

    concat_list = os.path.join(tmp, "concat.txt")
    with open(concat_list, "w") as f:
        for path, dur in all_files:
            f.write(f"file '{path}'\nduration {dur}\n")

    print("  Concatenating segments...")
    concat_raw = os.path.join(tmp, "concat_raw.mp4")
    _run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_list, "-c", "copy", concat_raw,
    ])

    # ── Pre-loop music ─────────────────────────────────────────────────────────
    looped_music: str | None = None
    if music_path:
        looped_music = os.path.join(tmp, "music_loop.wav")
        _run([
            "ffmpeg", "-y", "-stream_loop", "-1", "-i", music_path,
            "-t", str(total_dur),
            "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "2",
            looped_music,
        ])

    # ── Compute whoosh delay times ─────────────────────────────────────────────
    # Whoosh fires WHOOSH_DUR before the clip→transition cut,
    # i.e. at (cumulative_time_at_clip_end - WHOOSH_DUR).
    # There is one whoosh per clip except the last (which fades into outro).
    whoosh_delays_ms: list[int] = []
    cursor = INTRO_DUR
    for i, (trans_path, _, clip_dur) in enumerate(segments):
        cursor += clip_dur
        if trans_path is not None:   # there is a transition after this clip
            delay_ms = int((cursor - WHOOSH_DUR) * 1000)
            whoosh_delays_ms.append(max(0, delay_ms))
        # advance past the transition screen (if any)
        if trans_path is not None:
            cursor += TRANS_DUR

    n_whooshes = len(whoosh_delays_ms)

    # ── Build audio filter_complex ─────────────────────────────────────────────
    inputs = ["-i", concat_raw]
    for _ in range(n_whooshes):
        inputs += ["-i", whoosh_path]
    if looped_music:
        inputs += ["-i", looped_music]

    filter_parts: list[str] = []
    for i, delay_ms in enumerate(whoosh_delays_ms):
        filter_parts.append(f"[{i + 1}:a]adelay={delay_ms}|{delay_ms}[w{i}]")

    if looped_music:
        mi = 1 + n_whooshes
        filter_parts.append(f"[{mi}:a]volume={MUSIC_VOL}[bg]")

    whoosh_labels = "".join(f"[w{i}]" for i in range(n_whooshes))
    bg_label = "[bg]" if looped_music else ""
    n_mix = 1 + n_whooshes + (1 if looped_music else 0)
    filter_parts.append(
        f"[0:a]{whoosh_labels}{bg_label}"
        f"amix=inputs={n_mix}:duration=first:dropout_transition=0[aout]"
    )

    print("  Mixing audio and exporting...")
    _run([
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", ";".join(filter_parts),
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        output_path,
    ], silent=False)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Create a ranked YouTube Short.")
    parser.add_argument("input", nargs="?",
        help="YouTube URL or local video file (Mode 1). Omit for clips.txt (Mode 2).")
    parser.add_argument("--clips",  default="clips.txt", metavar="FILE")
    parser.add_argument("--music",  default="music.mp3",  metavar="FILE")
    parser.add_argument("--output", default="output.mp4", metavar="FILE")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:

        # ── Acquire sources ────────────────────────────────────────────────────
        # Each source: (video_path, timestamps, scores, fps, duration)
        raw_sources: list[tuple[str, list, list, float, float]] = []

        if args.input:
            print("Mode 1: Auto-clip from single video")
            if args.input.startswith("http"):
                src = os.path.join(tmp, "source.mp4")
                download_video(args.input, src)
            else:
                src = args.input
                if not os.path.exists(src):
                    sys.exit(f"Error: not found: {src}")

            print("  Analysing motion...")
            ts, sc, fps, dur = analyse_video(src)
            moments = find_top_moments(ts, sc, dur, NUM_CLIPS)
            for t in moments:
                raw_sources.append((src, ts, sc, fps, dur, t))

        else:
            print("Mode 2: Using clips from clips.txt")
            if not os.path.exists(args.clips):
                sys.exit(f"Error: {args.clips} not found.")
            with open(args.clips) as fh:
                entries = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
            if not entries:
                sys.exit("Error: clips.txt is empty.")
            entries = entries[:NUM_CLIPS]

            for i, entry in enumerate(entries):
                print(f"\nClip {i + 1}/{len(entries)}: {entry}")
                if entry.startswith("http"):
                    path = os.path.join(tmp, f"src_{i}.mp4")
                    download_video(entry, path)
                else:
                    path = entry
                    if not os.path.exists(path):
                        print("  Warning: not found, skipping")
                        continue
                ts, sc, fps, dur = analyse_video(path)
                moments = find_top_moments(ts, sc, dur, n=1)
                raw_sources.append((path, ts, sc, fps, dur, moments[0]))

        # raw_sources elements: (path, ts, sc, fps, dur, center_time)
        if not raw_sources:
            sys.exit("Error: no valid clips found.")

        n = len(raw_sources)

        # ── Process clips (rank n → 1) ─────────────────────────────────────────
        print("\nProcessing clips...")
        processed: list[tuple[str, float]] = []   # (path, actual_dur)

        for i, item in enumerate(raw_sources):
            path, ts, sc, fps, dur, center = item
            rank = n - i   # 5, 4, 3, 2, 1

            s, e = natural_clip_bounds(ts, sc, center, dur)
            actual_dur = round(e - s, 3)
            print(f"  Clip {i + 1}/{n} → Rank #{rank}  "
                  f"t={s:.1f}–{e:.1f}s  ({actual_dur:.1f}s)")

            cl_path = os.path.join(tmp, f"clip_{i}.mp4")
            process_clip(path, s, e, rank, cl_path)
            processed.append((cl_path, actual_dur))

        # ── Generate auxiliary clips ───────────────────────────────────────────
        print("Generating intro / transitions / outro...")
        intro_path = os.path.join(tmp, "intro.mp4")
        make_intro(intro_path)

        outro_path = os.path.join(tmp, "outro.mp4")
        make_outro(outro_path)

        # Build segment list with transitions between clips
        # Structure: clip#5 (no leading trans), then trans→clip for #4..#1
        segments: list[tuple[str | None, str, float]] = []
        for i, (cl_path, cl_dur) in enumerate(processed):
            rank_next = n - i - 1   # rank of the NEXT clip
            if i < n - 1:
                tr_path = os.path.join(tmp, f"trans_{i}.mp4")
                make_transition(rank_next, tr_path)
                trans = tr_path
            else:
                trans = None   # no transition after the last clip
            segments.append((trans, cl_path, cl_dur))

        # ── Whoosh and music ───────────────────────────────────────────────────
        whoosh_path = os.path.join(tmp, "whoosh.wav")
        make_whoosh(whoosh_path)

        music_path = args.music if os.path.exists(args.music) else None
        if not music_path:
            tag = "no music.mp3" if args.music == "music.mp3" else f"{args.music} not found"
            print(f"Note: {tag} — skipping background music")
        else:
            print(f"Music: {args.music} at {int(MUSIC_VOL*100)}% volume")

        # ── Assemble ───────────────────────────────────────────────────────────
        print("\nAssembling final video...")
        assemble(
            intro_path, segments, outro_path,
            whoosh_path, music_path, args.output, tmp,
        )
        print(f"\nDone!  →  {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
