#!/usr/bin/env python3
"""
make_short.py - Creates a professional ranked YouTube Short from video clips.

Mode 1 (auto): Pass a YouTube URL or local video file.
               Finds the 5 most interesting moments via motion detection.

Mode 2 (manual): Omit the argument; reads up to 5 URLs/paths from clips.txt.
                 Finds the best 5-second moment in each clip.

Output structure:
  [1s TOP 5 intro] → [0.4s countdown] → [5s clip #5] → ... → [5s clip #1]

Requires: yt-dlp, ffmpeg (system), opencv-python, numpy
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
CLIP_DURATION = 5.0
INTRO_DUR    = 1.0    # "TOP 5" hook screen
COUNTDOWN_DUR = 0.4   # rank flash before each clip
FADE_DUR     = 0.3    # fade-to-black at end of each clip
WHOOSH_DUR   = 0.45
NUM_CLIPS    = 5

SEGMENT_DUR  = COUNTDOWN_DUR + CLIP_DURATION   # 5.4 s per rank slot

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


def _escape_fontpath(path: str) -> str:
    return path.replace("\\", "/").replace(":", "\\:")


def _font_opt() -> str:
    fp = _find_font()
    return f":fontfile='{_escape_fontpath(fp)}'" if fp else ""


# ── Download ───────────────────────────────────────────────────────────────────

def download_video(url: str, dest: str) -> None:
    print(f"  Downloading: {url}")
    subprocess.run([
        sys.executable, "-m", "yt_dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", dest, url,
    ], check=True)
    print("  Download complete.")


# ── Motion analysis ────────────────────────────────────────────────────────────

def _motion_scores(video_path: str):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(1, int(fps / 4))
    scores, prev_gray, idx = [], None, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % interval == 0:
            small = cv2.resize(frame, (320, 240))
            gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (21, 21), 0)
            if prev_gray is not None:
                scores.append((idx / fps, float(np.mean(cv2.absdiff(prev_gray, gray)))))
            prev_gray = gray
        idx += 1

    cap.release()
    return scores, fps


def find_top_moments(video_path: str, n: int = NUM_CLIPS) -> list:
    """Return n start-timestamps (sec) of highest-motion moments, min-spaced."""
    scores, _ = _motion_scores(video_path)

    cap = cv2.VideoCapture(video_path)
    duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / (cap.get(cv2.CAP_PROP_FPS) or 30)
    cap.release()

    if not scores:
        step = duration / (n + 1)
        return [step * (i + 1) for i in range(n)]

    vals = np.array([s for _, s in scores])
    window = min(16, max(1, len(vals) // 10))
    if window > 1:
        vals = np.convolve(vals, np.ones(window) / window, mode="same")

    timestamps = [t for t, _ in scores]
    min_gap = CLIP_DURATION + 2.0
    selected = []

    for idx in np.argsort(vals)[::-1]:
        t = timestamps[idx]
        # Clamp so a 5-second clip starting at t fits inside the video
        t = max(0.0, min(duration - CLIP_DURATION, t))
        if not any(abs(t - s) < min_gap for s in selected):
            selected.append(t)
        if len(selected) == n:
            break

    while len(selected) < n:
        step = duration / (n + 1)
        selected.append(step * (len(selected) + 1))

    selected.sort()
    print(f"  Key moments: {[f'{t:.1f}s' for t in selected[:n]]}")
    return selected[:n]


def find_best_moment(video_path: str) -> float:
    return find_top_moments(video_path, n=1)[0]


# ── Black-screen title clips ───────────────────────────────────────────────────

def _black_clip(duration: float, vf: str, output_path: str) -> None:
    """Render a black-background title/countdown clip with the given vf chain."""
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=black:size={TARGET_W}x{TARGET_H}:rate=30",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-t", str(duration),
        "-vf", vf,
        "-filter_complex", "[1:a]aresample=44100[audio]",
        "-map", "0:v", "-map", "[audio]",
        "-c:v", "libx264", "-crf", "23", "-preset", "medium",
        "-c:a", "aac", "-ar", "44100", "-ac", "2",
        output_path,
    ], check=True, stderr=subprocess.DEVNULL)


def generate_intro_clip(output_path: str) -> None:
    """1-second 'TOP 5' hook screen."""
    fo = _font_opt()
    vf = (
        f"drawtext=text='TOP 5'{fo}:fontsize=300"
        f":fontcolor=white:borderw=22:bordercolor=black"
        f":x=(w-text_w)/2:y=(h-text_h)/2"
    )
    _black_clip(INTRO_DUR, vf, output_path)


def generate_countdown_clip(rank: int, output_path: str) -> None:
    """0.4-second rank-number flash on black — the countdown beat."""
    fo = _font_opt()
    vf = (
        f"drawtext=text='{rank}'{fo}:fontsize=400"
        f":fontcolor=white:borderw=28:bordercolor=black"
        f":x=(w-text_w)/2:y=(h-text_h)/2"
    )
    _black_clip(COUNTDOWN_DUR, vf, output_path)


# ── Per-clip processing ────────────────────────────────────────────────────────

def process_clip_ffmpeg(source_path: str, start_time: float, rank: int, output_path: str) -> None:
    """
    Extract 5-second clip starting at start_time, crop to 9:16,
    burn rank overlay + 'RANKED DAILY' label, fade to black at end.
    """
    t_start = max(0.0, start_time)

    scale_crop = (
        f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=increase,"
        f"crop={TARGET_W}:{TARGET_H}"
    )

    fo = _font_opt()

    # Rank number: top-left, gold drop-shadow + white with thick black border
    dt_shadow = (
        f"drawtext=text='{rank}'{fo}:fontsize=180"
        f":fontcolor='0xFFD700@0.85':x=44:y=44"
    )
    dt_rank = (
        f"drawtext=text='{rank}'{fo}:fontsize=180"
        f":fontcolor=white:borderw=14:bordercolor=black:x=40:y=40"
    )

    # "RANKED DAILY" small-caps label at bottom
    dt_label = (
        f"drawtext=text='RANKED DAILY'{fo}:fontsize=52"
        f":fontcolor=white:borderw=4:bordercolor=black"
        f":x=(w-text_w)/2:y=h-100"
    )

    # Fade to black at end
    fade = f"fade=t=out:st={CLIP_DURATION - FADE_DUR}:d={FADE_DUR}"

    vf = f"{scale_crop},{dt_shadow},{dt_rank},{dt_label},{fade}"
    af = f"afade=t=out:st={CLIP_DURATION - FADE_DUR}:d={FADE_DUR}"

    if _has_audio(source_path):
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(t_start), "-i", source_path,
            "-t", str(CLIP_DURATION),
            "-vf", vf, "-af", af,
            "-map", "0:v", "-map", "0:a",
            "-c:v", "libx264", "-crf", "23", "-preset", "medium",
            "-c:a", "aac", "-ar", "44100", "-ac", "2",
            output_path,
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(t_start), "-i", source_path,
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
            "-t", str(CLIP_DURATION),
            "-vf", vf,
            "-filter_complex",
            f"[1:a]aresample=44100,afade=t=out:st={CLIP_DURATION - FADE_DUR}:d={FADE_DUR}[audio]",
            "-map", "0:v", "-map", "[audio]",
            "-c:v", "libx264", "-crf", "23", "-preset", "medium",
            "-c:a", "aac", "-ar", "44100", "-ac", "2",
            output_path,
        ]

    subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)


# ── Whoosh sound ───────────────────────────────────────────────────────────────

def generate_whoosh_wav(output_path: str, sr: int = 44100) -> None:
    """Synthesise a high→low whoosh and write as 16-bit mono WAV."""
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
    with wave.open(output_path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


# ── Final assembly ─────────────────────────────────────────────────────────────

def build_final_video(
    intro_path: str,
    countdown_paths: list,
    clip_paths: list,
    whoosh_path: str,
    music_path: str | None,
    output_path: str,
    tmp: str,
) -> None:
    """
    Assemble: intro → [countdown + clip] × N
    Mix in whoosh transitions and background music, export final MP4.
    """
    n_clips    = len(clip_paths)
    n_whooshes = n_clips - 1
    total_dur  = INTRO_DUR + n_clips * SEGMENT_DUR

    # ── Concatenate all segments ───────────────────────────────────────────────
    print("  Concatenating segments...")
    concat_list = os.path.join(tmp, "concat.txt")
    with open(concat_list, "w") as f:
        f.write(f"file '{intro_path}'\nduration {INTRO_DUR}\n")
        for cd, cp in zip(countdown_paths, clip_paths):
            f.write(f"file '{cd}'\nduration {COUNTDOWN_DUR}\n")
            f.write(f"file '{cp}'\nduration {CLIP_DURATION}\n")

    concat_raw = os.path.join(tmp, "concat_raw.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_list, "-c", "copy", concat_raw,
    ], check=True, stderr=subprocess.DEVNULL)

    # ── Pre-loop music ─────────────────────────────────────────────────────────
    looped_music: str | None = None
    if music_path:
        looped_music = os.path.join(tmp, "music_looped.wav")
        subprocess.run([
            "ffmpeg", "-y", "-stream_loop", "-1", "-i", music_path,
            "-t", str(total_dur),
            "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "2",
            looped_music,
        ], check=True, stderr=subprocess.DEVNULL)

    # ── Audio filter_complex: whooshes + background music ─────────────────────
    #
    # Input layout:
    #   [0]             concat_raw.mp4
    #   [1 … n_whooshes]  whoosh.wav  (one per transition)
    #   [n_whooshes+1]  music_looped.wav  (optional)
    #
    # Whoosh i fires WHOOSH_DUR before the cut after clip i:
    #   delay = INTRO_DUR + (i+1)*SEGMENT_DUR - WHOOSH_DUR

    inputs = ["-i", concat_raw]
    for _ in range(n_whooshes):
        inputs += ["-i", whoosh_path]
    if looped_music:
        inputs += ["-i", looped_music]

    filter_parts = []
    for i in range(n_whooshes):
        delay_ms = int((INTRO_DUR + (i + 1) * SEGMENT_DUR - WHOOSH_DUR) * 1000)
        filter_parts.append(f"[{i + 1}:a]adelay={delay_ms}|{delay_ms}[w{i}]")

    if looped_music:
        mi = 1 + n_whooshes
        filter_parts.append(f"[{mi}:a]volume=0.12[bg]")

    whoosh_labels = "".join(f"[w{i}]" for i in range(n_whooshes))
    bg_label = "[bg]" if looped_music else ""
    n_mix = 1 + n_whooshes + (1 if looped_music else 0)
    filter_parts.append(
        f"[0:a]{whoosh_labels}{bg_label}"
        f"amix=inputs={n_mix}:duration=first:dropout_transition=0[aout]"
    )

    print("  Mixing audio and exporting...")
    subprocess.run([
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", ";".join(filter_parts),
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        output_path,
    ], check=True)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Create a ranked YouTube Short.")
    parser.add_argument("input", nargs="?",
        help="YouTube URL or local video file (Mode 1). Omit for clips.txt (Mode 2).")
    parser.add_argument("--clips", default="clips.txt", metavar="FILE")
    parser.add_argument("--music", default="music.mp3", metavar="FILE")
    parser.add_argument("--output", default="output.mp4", metavar="FILE")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        sources: list[tuple[str, float]] = []

        # ── Source acquisition ─────────────────────────────────────────────────
        if args.input:
            print("Mode 1: Auto-clip from single video")
            if args.input.startswith("http"):
                src_path = os.path.join(tmp, "source.mp4")
                download_video(args.input, src_path)
            else:
                src_path = args.input
                if not os.path.exists(src_path):
                    sys.exit(f"Error: File not found: {src_path}")

            print("Analysing video for interesting moments...")
            timestamps = find_top_moments(src_path, NUM_CLIPS)
            sources = [(src_path, t) for t in timestamps]

        else:
            print("Mode 2: Using clips from clips.txt")
            if not os.path.exists(args.clips):
                sys.exit(f"Error: {args.clips} not found.")

            with open(args.clips) as fh:
                entries = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]

            if not entries:
                sys.exit("Error: No entries found in clips.txt")

            if len(entries) > NUM_CLIPS:
                print(f"  Note: using first {NUM_CLIPS} of {len(entries)} entries")
                entries = entries[:NUM_CLIPS]

            for i, entry in enumerate(entries):
                print(f"\nClip {i + 1}/{len(entries)}: {entry}")
                if entry.startswith("http"):
                    path = os.path.join(tmp, f"clip_{i}.mp4")
                    download_video(entry, path)
                else:
                    path = entry
                    if not os.path.exists(path):
                        print("  Warning: not found, skipping")
                        continue
                sources.append((path, find_best_moment(path)))

        if not sources:
            sys.exit("Error: No valid clips to process.")

        n = len(sources)

        # ── Build title intro ──────────────────────────────────────────────────
        print("\nGenerating intro screen...")
        intro_path = os.path.join(tmp, "intro.mp4")
        generate_intro_clip(intro_path)

        # ── Process clips + countdown screens ─────────────────────────────────
        # sources are sorted chronologically; rank 5 = index 0, rank 1 = index n-1
        print("\nProcessing clips...")
        clip_paths = []
        countdown_paths = []

        for i, (path, t) in enumerate(sources):
            rank = n - i   # 5, 4, 3, 2, 1
            print(f"  Clip {i + 1}/{n} → Rank #{rank}  (t={t:.1f}s)")

            cd_path = os.path.join(tmp, f"countdown_{i}.mp4")
            generate_countdown_clip(rank, cd_path)
            countdown_paths.append(cd_path)

            cl_path = os.path.join(tmp, f"processed_{i}.mp4")
            process_clip_ffmpeg(path, t, rank, cl_path)
            clip_paths.append(cl_path)

        # ── Whoosh + music ─────────────────────────────────────────────────────
        whoosh_path = os.path.join(tmp, "whoosh.wav")
        generate_whoosh_wav(whoosh_path)

        music_path = args.music if os.path.exists(args.music) else None
        if not music_path:
            tag = "no music.mp3 found" if args.music == "music.mp3" else f"{args.music} not found"
            print(f"Note: {tag} — skipping background music")
        else:
            print(f"Background music: {args.music} (12% volume)")

        # ── Assemble final video ───────────────────────────────────────────────
        print("\nBuilding final video...")
        build_final_video(
            intro_path, countdown_paths, clip_paths,
            whoosh_path, music_path, args.output, tmp,
        )
        print(f"\nDone!  Saved to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
