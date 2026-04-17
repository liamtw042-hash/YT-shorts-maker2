#!/usr/bin/env python3
"""
make_short.py - Creates a ranked YouTube Short from video clips.

Mode 1 (auto): Pass a YouTube URL or local video file.
               Finds the 5 most interesting moments via motion detection.

Mode 2 (manual): Omit the argument; reads up to 5 URLs/paths from clips.txt.
                 Finds the best 5-second moment in each clip.

Both modes produce a 9:16 vertical Short with countdown overlays (5→1),
whoosh transitions, optional background music, and exports as output.mp4.
"""

import os
import sys
import argparse
import tempfile
import subprocess
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from moviepy.editor import (
    VideoFileClip,
    ImageClip,
    CompositeVideoClip,
    AudioFileClip,
    concatenate_videoclips,
    CompositeAudioClip,
)
from moviepy.audio.AudioClip import AudioArrayClip, concatenate_audioclips


# ── Constants ──────────────────────────────────────────────────────────────────

TARGET_W = 1080
TARGET_H = 1920
CLIP_DURATION = 5.0
NUM_CLIPS = 5

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "C:/Windows/Fonts/arialbd.ttf",
]


# ── Download ───────────────────────────────────────────────────────────────────

def download_video(url: str, dest: str) -> None:
    print(f"  Downloading: {url}")
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", dest,
        url,
    ]
    subprocess.run(cmd, check=True)
    print("  Download complete.")


# ── Motion analysis ────────────────────────────────────────────────────────────

def _motion_scores(video_path: str):
    """Return [(timestamp_sec, motion_score), ...] sampled at ~4 fps."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(1, int(fps / 4))

    scores = []
    prev_gray = None
    idx = 0

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
    """Return n timestamps (sec) of highest-motion moments with min spacing."""
    scores, fps = _motion_scores(video_path)

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
        half = CLIP_DURATION / 2
        t = max(half, min(duration - half, t))
        if not any(abs(t - s) < min_gap for s in selected):
            selected.append(t)
        if len(selected) == n:
            break

    # Fill gaps if video is too short to yield n well-spaced moments
    while len(selected) < n:
        step = duration / (n + 1)
        candidate = step * (len(selected) + 1)
        selected.append(candidate)

    selected.sort()
    print(f"  Key moments: {[f'{t:.1f}s' for t in selected[:n]]}")
    return selected[:n]


def find_best_moment(video_path: str) -> float:
    return find_top_moments(video_path, n=1)[0]


# ── Video transform ────────────────────────────────────────────────────────────

def crop_to_vertical(clip):
    """Centre-crop then resize to TARGET_W × TARGET_H."""
    w, h = clip.size
    want_ratio = TARGET_W / TARGET_H
    have_ratio = w / h

    if have_ratio > want_ratio:          # too wide → crop sides
        new_w = int(h * want_ratio)
        x = (w - new_w) // 2
        clip = clip.crop(x1=x, x2=x + new_w)
    elif have_ratio < want_ratio:        # too tall → crop top/bottom
        new_h = int(w / want_ratio)
        y = (h - new_h) // 2
        clip = clip.crop(y1=y, y2=y + new_h)

    return clip.resize((TARGET_W, TARGET_H))


# ── Rank overlay ───────────────────────────────────────────────────────────────

def _load_font(size: int):
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def make_rank_overlay(rank: int, duration: float):
    """
    Big bold rank number, white with black outline + gold drop-shadow.
    Displayed for the first 2.5 s of the clip, then disappears.
    """
    img = Image.new("RGBA", (TARGET_W, TARGET_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = _load_font(350)
    text = str(rank)

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (TARGET_W - tw) // 2 - bbox[0]
    y = (TARGET_H - th) // 2 - bbox[1]

    # Thick black outline
    for dx in range(-14, 15, 4):
        for dy in range(-14, 15, 4):
            if dx * dx + dy * dy <= 196:
                draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0, 220))

    # Gold drop-shadow
    draw.text((x + 8, y + 8), text, font=font, fill=(200, 160, 0, 180))

    # White fill
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))

    arr = np.array(img)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3].astype(float) / 255.0

    show_for = min(2.5, duration)
    text_clip = (
        ImageClip(rgb, ismask=False)
        .set_mask(ImageClip(alpha, ismask=True).set_duration(show_for))
        .set_duration(show_for)
        .fadeout(0.4)
    )
    return text_clip


# ── Audio effects ──────────────────────────────────────────────────────────────

def make_whoosh(duration: float = 0.45) -> AudioArrayClip:
    """Synthesise a high→low frequency whoosh with noise texture."""
    sr = 44100
    n = int(sr * duration)
    t = np.linspace(0, duration, n, dtype=np.float32)

    # Exponential frequency sweep 1 400 Hz → 80 Hz
    freqs = 1400.0 * np.exp(-5.0 * t / duration) + 80.0
    phase = 2 * np.pi * np.cumsum(freqs) / sr
    tone = np.sin(phase)

    noise = np.convolve(
        np.random.normal(0, 1.0, n).astype(np.float32),
        np.ones(30, dtype=np.float32) / 30,
        mode="same",
    )

    wave = tone * 0.55 + noise * 0.45
    env = np.exp(-5.0 * t / duration)
    env[: int(0.03 * sr)] *= np.linspace(0, 1, int(0.03 * sr), dtype=np.float32)
    wave = (wave * env * 0.8).astype(np.float32)

    stereo = np.column_stack([wave, wave])
    return AudioArrayClip(stereo, fps=sr)


# ── Clip builder ───────────────────────────────────────────────────────────────

def build_clip(video_path: str, center_time: float, rank: int):
    """Extract 5-second clip centred on center_time, add vertical crop + rank overlay."""
    src = VideoFileClip(video_path)
    half = CLIP_DURATION / 2
    t0 = max(0.0, center_time - half)
    t1 = min(src.duration, t0 + CLIP_DURATION)
    t0 = max(0.0, t1 - CLIP_DURATION)

    clip = src.subclip(t0, t1)
    clip = crop_to_vertical(clip)
    overlay = make_rank_overlay(rank, clip.duration)
    clip = CompositeVideoClip([clip, overlay], use_bgclip=True)
    src.close()
    return clip


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Create a ranked YouTube Short.")
    parser.add_argument(
        "input", nargs="?",
        help="YouTube URL or local video file (Mode 1). Omit to use clips.txt (Mode 2).",
    )
    parser.add_argument("--clips", default="clips.txt", metavar="FILE",
                        help="Clips list file for Mode 2 (default: clips.txt)")
    parser.add_argument("--music", default="music.mp3", metavar="FILE",
                        help="Background music file (default: music.mp3)")
    parser.add_argument("--output", default="output.mp4", metavar="FILE",
                        help="Output file (default: output.mp4)")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        sources = []   # [(video_path, center_time_sec)]

        # ── Mode 1: single source, auto-detect moments ─────────────────────────
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

        # ── Mode 2: per-clip URLs from clips.txt ───────────────────────────────
        else:
            print("Mode 2: Using clips from clips.txt")
            if not os.path.exists(args.clips):
                sys.exit(f"Error: {args.clips} not found. "
                         "Create it with one URL or file path per line.")

            with open(args.clips) as fh:
                entries = [
                    ln.strip()
                    for ln in fh
                    if ln.strip() and not ln.startswith("#")
                ]

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
                        print(f"  Warning: not found, skipping")
                        continue
                t = find_best_moment(path)
                sources.append((path, t))

        if not sources:
            sys.exit("Error: No valid clips to process.")

        # ── Build ranked clips ─────────────────────────────────────────────────
        print("\nBuilding clips...")
        n = len(sources)
        clips = []
        for i, (path, t) in enumerate(sources):
            rank = n - i          # 5, 4, 3, 2, 1
            print(f"  Clip {i + 1}/{n} → Rank #{rank}  (source t={t:.1f}s)")
            clips.append(build_clip(path, t, rank))

        # ── Whoosh transitions ─────────────────────────────────────────────────
        print("Adding whoosh transitions...")
        whoosh = make_whoosh()

        final_clips = []
        for i, clip in enumerate(clips):
            if i < len(clips) - 1:
                ws_start = max(0.0, clip.duration - whoosh.duration)
                ws = whoosh.set_start(ws_start)
                orig_audio = clip.audio
                clip = clip.set_audio(
                    CompositeAudioClip([orig_audio, ws]) if orig_audio else ws
                )
            final_clips.append(clip)

        # ── Concatenate ────────────────────────────────────────────────────────
        print("Concatenating clips...")
        video = concatenate_videoclips(final_clips, method="compose")

        # ── Background music ───────────────────────────────────────────────────
        if os.path.exists(args.music):
            print(f"Mixing background music from {args.music}...")
            bg = AudioFileClip(args.music)
            if bg.duration < video.duration:
                loops = int(np.ceil(video.duration / bg.duration))
                bg = concatenate_audioclips([AudioFileClip(args.music)] * loops)
            bg = bg.subclip(0, video.duration).volumex(0.15)
            existing = video.audio
            video = video.set_audio(
                CompositeAudioClip([existing, bg]) if existing else bg
            )
        else:
            if args.music != "music.mp3":
                print(f"Warning: {args.music} not found — skipping background music")
            else:
                print("Note: no music.mp3 found — skipping background music")

        # ── Export ─────────────────────────────────────────────────────────────
        print(f"\nExporting → {args.output}")
        video.write_videofile(
            args.output,
            fps=30,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=os.path.join(tmp, "tmp_audio.m4a"),
            remove_temp=True,
            preset="medium",
            ffmpeg_params=["-crf", "23"],
            logger="bar",
        )

        print(f"\nDone!  Saved to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
