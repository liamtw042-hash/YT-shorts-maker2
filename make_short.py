#!/usr/bin/env python3
"""
make_short.py — RankZilla-style Short from a single YouTube source video.

Downloads one video, auto-detects 5 distinct high-motion clips,
removes existing overlays via crop, rebuilds in RankZilla layout.

Usage:
    python make_short.py "https://youtube.com/shorts/xxx" \
        --title "Ranking Funniest" --subtitle "Memes 2026"

Output: output.mp4 (1080x1920, 30fps, h264/aac)
Requires: ffmpeg (system), yt-dlp (pip), opencv-python (pip), numpy (pip)
"""

import json
import os
import shutil
import sys
import time
import argparse
import tempfile
import subprocess

import cv2
import numpy as np


# ── Constants ──────────────────────────────────────────────────────────────────

TARGET_W  = 1080
TARGET_H  = 1920
FPS       = 30
BANNER_H  = 120
NUM_CLIPS = 5
MAX_RETRY = 3
MIN_WIN   = 5.0   # seconds
MAX_WIN   = 8.0   # seconds

# Left-sidebar rank-number vertical centres (rank 1 top → rank 5 bottom).
# Usable area below banner: y=120–1920 (1800 px), 5 slots 200 px apart.
RANK_Y = {r: 1020 + (r - 3) * 200 for r in range(1, 6)}
# {1:620, 2:820, 3:1020, 4:1220, 5:1420}

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

def _find_font() -> str | None:
    for fp in FONT_CANDIDATES:
        if os.path.exists(fp):
            return fp
    return None


def _fo() -> str:
    fp = _find_font()
    if not fp:
        return ""
    safe = fp.replace("\\", "/").replace(":", "\\:")
    return f":fontfile='{safe}'"


def _esc(text: str) -> str:
    """Escape text for use inside ffmpeg drawtext=text='...'"""
    return (text
            .replace("\\", "\\\\")
            .replace("'",  "\\'")
            .replace(":",  "\\:")
            .replace("%",  "\\%"))


def _has_audio(path: str) -> bool:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-select_streams", "a", path],
            capture_output=True, text=True, timeout=30,
        )
        return bool(json.loads(r.stdout).get("streams"))
    except Exception:
        return False


def _run(cmd: list, label: str = "") -> None:
    """Run a command. On non-zero exit, raise RuntimeError with full stderr."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tag = f" [{label}]" if label else ""
        raise RuntimeError(
            f"Command failed{tag} (exit {result.returncode})\n"
            f"CMD: {' '.join(str(x) for x in cmd[:8])} ...\n"
            f"STDERR (last 4000 chars):\n{result.stderr[-4000:]}"
        )


def _verify(path: str, label: str = "") -> None:
    """Raise if path is missing or empty."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise RuntimeError(
            f"Expected output missing or empty"
            f"{' [' + label + ']' if label else ''}: {path}"
        )


# ── Download ───────────────────────────────────────────────────────────────────

def download(url: str, dest: str) -> None:
    """Download with up to MAX_RETRY attempts. sys.exit on permanent failure."""
    for attempt in range(1, MAX_RETRY + 1):
        print(f"  [{attempt}/{MAX_RETRY}] {url}")
        try:
            subprocess.run(
                [
                    sys.executable, "-m", "yt_dlp",
                    "--merge-output-format", "mp4",
                    "--format",
                    "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
                    "--retries", "10",
                    "--postprocessor-args", "-vcodec libx264 -acodec aac",
                    "-o", dest,
                    url,
                ],
                check=True,
                timeout=600,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            print(f"  Attempt {attempt} failed: {exc}")
            if attempt < MAX_RETRY:
                wait = 4 * attempt
                print(f"  Retrying in {wait}s ...")
                time.sleep(wait)
                continue
            sys.exit(
                f"\nERROR: Download failed after {MAX_RETRY} attempts.\n"
                f"URL: {url}\nLast error: {exc}"
            )

        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"  OK — {os.path.getsize(dest) // 1024} KB")
            return

        print(f"  File missing or empty after attempt {attempt}.")
        if attempt == MAX_RETRY:
            sys.exit(f"\nERROR: Download produced empty file: {dest}")
        time.sleep(4 * attempt)


# ── Motion analysis ────────────────────────────────────────────────────────────

def analyse_video(path: str) -> tuple[list, list, float, float]:
    """Frame-diff motion at ~4 fps. Returns (timestamps, scores, fps, duration)."""
    cap  = cv2.VideoCapture(path)
    fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tot  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur  = tot / fps if tot > 0 else 0.0
    step = max(1, int(fps / 4))

    ts, sc, prev, idx = [], [], None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            sm   = cv2.resize(frame, (320, 240))
            gray = cv2.GaussianBlur(
                cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY), (21, 21), 0
            )
            if prev is not None:
                ts.append(idx / fps)
                sc.append(float(np.mean(cv2.absdiff(prev, gray))))
            prev = gray
        idx += 1

    cap.release()
    return ts, sc, fps, dur


def _natural_window(ts: np.ndarray, sc: np.ndarray,
                    pi: int, duration: float) -> tuple[float, float]:
    """Expand from peak index pi until motion < 30% of peak, capped at MAX_WIN."""
    thresh = float(sc[pi]) * 0.30
    sps    = len(ts) / duration if duration > 0 else 4.0
    walk   = max(4, int(MAX_WIN * sps))

    s = float(ts[pi])
    for i in range(pi - 1, max(0, pi - walk), -1):
        if sc[i] < thresh:
            s = float(ts[i + 1] if i + 1 < len(ts) else ts[i])
            break
        s = float(ts[i])

    e = float(ts[pi])
    for i in range(pi + 1, min(len(ts), pi + walk)):
        if sc[i] < thresh:
            e = float(ts[i - 1] if i > 0 else ts[i])
            break
        e = float(ts[i])

    d = e - s
    if d < MIN_WIN:
        e = min(duration, s + MIN_WIN)
        if e - s < MIN_WIN:          # not enough room forward, pull start back
            s = max(0.0, e - MIN_WIN)
    elif d > MAX_WIN:
        mid = (s + e) / 2.0
        s, e = mid - MAX_WIN / 2, mid + MAX_WIN / 2

    return max(0.0, s), min(duration, e)


def find_5_windows(timestamps: list, scores: list,
                   duration: float) -> list[dict]:
    """
    Find 5 distinct high-motion windows in a single video.
    Returns list of dicts sorted by excitement, most exciting first (= rank #1).
    """
    if not timestamps or len(timestamps) < NUM_CLIPS:
        step = duration / (NUM_CLIPS + 1)
        return [
            {"start": max(0.0, step * (i + 1) - MIN_WIN / 2),
             "end":   min(duration, step * (i + 1) + MIN_WIN / 2),
             "score": 1.0}
            for i in range(NUM_CLIPS)
        ]

    ts   = np.array(timestamps)
    sc   = np.array(scores)
    w    = min(16, max(1, len(sc) // 10))
    sc_s = np.convolve(sc, np.ones(w) / w, mode="same") if w > 1 else sc.copy()
    sps  = len(ts) / duration  # samples per second

    # Find 5 well-separated peaks; relax gap until we have enough
    peaks: list[int] = []
    for gap_sec in [MAX_WIN + 2, MAX_WIN, MIN_WIN + 2, MIN_WIN, 2.0, 0.5]:
        gap = max(2, int(gap_sec * sps))
        peaks = []
        for pi in np.argsort(sc_s)[::-1]:
            pi = int(pi)
            if not any(abs(pi - p) < gap for p in peaks):
                peaks.append(pi)
            if len(peaks) == NUM_CLIPS:
                break
        if len(peaks) == NUM_CLIPS:
            break

    # Last-resort fill with evenly spaced indices
    seen = set(peaks)
    for i in range(1, NUM_CLIPS * 4):
        if len(peaks) >= NUM_CLIPS:
            break
        idx = min(int(len(ts) * i / (NUM_CLIPS * 2)), len(ts) - 1)
        if idx not in seen:
            peaks.append(idx)
            seen.add(idx)

    peaks = peaks[:NUM_CLIPS]

    windows: list[dict] = []
    for pi in peaks:
        s, e = _natural_window(ts, sc_s, pi, duration)
        mask = (ts >= s) & (ts <= e)
        exc  = (float(np.percentile(sc[mask], 80))
                if mask.sum() > 0 else float(sc_s[pi]))
        windows.append({"start": s, "end": e, "score": exc})

    windows.sort(key=lambda x: x["score"], reverse=True)
    return windows


# ── Filter builders ────────────────────────────────────────────────────────────

def _banner_filters(w1: str, rest: str, subtitle: str, fo: str) -> list[str]:
    x_rest = 40 + int(len(w1) * 54 * 0.60) + 10
    parts = [
        f"drawbox=x=0:y=0:w={TARGET_W}:h={BANNER_H}:color=black:t=fill",
        f"drawtext=text='{_esc(w1)}'{fo}:fontsize=54"
        f":fontcolor=white:borderw=3:bordercolor=black:x=40:y=18",
    ]
    if rest:
        parts.append(
            f"drawtext=text='{_esc(rest)}'{fo}:fontsize=54"
            f":fontcolor='0xFF3333':borderw=3:bordercolor=black:x={x_rest}:y=18"
        )
    if subtitle:
        parts.append(
            f"drawtext=text='{_esc(subtitle)}'{fo}:fontsize=32"
            f":fontcolor=yellow:borderw=2:bordercolor=black:x=40:y=78"
        )
    return parts


def _rank_filters(active: int, fo: str) -> list[str]:
    parts = []
    for r in range(1, NUM_CLIPS + 1):
        cy = RANK_Y[r]
        if r == active:
            fs, col, bw, bc = 85, "yellow", 7, "black"
            y = cy - 42
        else:
            fs, col, bw, bc = 48, "'0xBBBBBB@0.60'", 2, "'0x00000066'"
            y = cy - 24
        parts.append(
            f"drawtext=text='{r}.'{fo}:fontsize={fs}:fontcolor={col}"
            f":borderw={bw}:bordercolor={bc}:x=30:y={y}"
        )
    return parts


def build_vf(active_rank: int, w1: str, rest: str,
             subtitle: str, fo: str) -> str:
    """
    Full -vf chain for one segment:
      1. Crop left 18% + top 12% (removes existing overlays)
      2. Letterbox back to 1080x1920
      3. Title banner + rank sidebar burned in
    """
    chain = [
        # Remove existing overlays
        "crop=iw*0.82:ih*0.88:iw*0.18:ih*0.12",
        # Letterbox to 1080x1920
        f"scale=w={TARGET_W}:h={TARGET_H}:force_original_aspect_ratio=decrease",
        f"pad={TARGET_W}:{TARGET_H}:(ow-iw)/2:(oh-ih)/2:black",
        f"fps={FPS}",
    ]
    chain += _banner_filters(w1, rest, subtitle, fo)
    chain += _rank_filters(active_rank, fo)
    return ",".join(chain)


# ── Segment processing ─────────────────────────────────────────────────────────

def process_segment(source: str, t_start: float, t_end: float,
                    active_rank: int, w1: str, rest: str, subtitle: str,
                    fo: str, out: str) -> float:
    """Render one clip window with overlays. Returns actual duration written."""
    dur = round(t_end - t_start, 3)
    vf  = build_vf(active_rank, w1, rest, subtitle, fo)
    af  = "loudnorm=I=-16:TP=-1.5:LRA=11"

    if _has_audio(source):
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(t_start), "-i", source,
            "-t", str(dur),
            "-vf", vf, "-af", af,
            "-map", "0:v", "-map", "0:a",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
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
            "-filter_complex", "[1:a]aresample=44100[a]",
            "-map", "0:v", "-map", "[a]",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac", "-ar", "44100", "-ac", "2",
            out,
        ]

    try:
        _run(cmd, f"segment rank#{active_rank}")
    except RuntimeError as exc:
        print(f"\nERROR in segment rank#{active_rank}:\n{exc}")
        raise

    _verify(out, f"segment rank#{active_rank}")
    return dur


# ── Assembly ───────────────────────────────────────────────────────────────────

def assemble(seg_files: list, seg_durs: list,
             music_path: str | None, output: str, tmp: str) -> None:
    """Hard-cut concat of segments; optionally mix music at 10% volume."""
    total_dur = sum(seg_durs)

    concat_txt = os.path.join(tmp, "concat.txt")
    with open(concat_txt, "w") as f:
        for sf, dur in zip(seg_files, seg_durs):
            f.write(f"file '{sf}'\nduration {dur}\n")

    concat_raw = os.path.join(tmp, "concat_raw.mp4")
    print("  Concatenating segments ...")
    try:
        _run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_txt, "-c", "copy", concat_raw,
        ], "concat")
    except RuntimeError as exc:
        print(f"\nERROR during concat:\n{exc}")
        raise
    _verify(concat_raw, "concat")

    if not music_path:
        shutil.copy2(concat_raw, output)
        return

    looped = os.path.join(tmp, "music_loop.wav")
    try:
        _run([
            "ffmpeg", "-y",
            "-stream_loop", "-1", "-i", music_path,
            "-t", str(total_dur),
            "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "2",
            looped,
        ], "music loop")
    except RuntimeError as exc:
        print(f"  WARNING: music loop failed — skipping.\n  {exc}")
        shutil.copy2(concat_raw, output)
        return

    print("  Mixing music and writing final file ...")
    try:
        _run([
            "ffmpeg", "-y",
            "-i", concat_raw, "-i", looped,
            "-filter_complex",
            "[1:a]volume=0.10[bg];"
            "[0:a][bg]amix=inputs=2:duration=first:dropout_transition=0[aout]",
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            output,
        ], "final mix")
    except RuntimeError as exc:
        print(f"\nERROR during final mix:\n{exc}")
        raise
    _verify(output, "final output")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RankZilla-style Short from one YouTube source video."
    )
    parser.add_argument("url",
        help="YouTube URL of the source video")
    parser.add_argument("--title", default="Ranking Top Moments",
        help="Title text — first word white, rest red (default: 'Ranking Top Moments')")
    parser.add_argument("--subtitle", default="",
        help="Yellow subtitle line below the title")
    parser.add_argument("--music",  default="music.mp3")
    parser.add_argument("--output", default="output.mp4")
    args = parser.parse_args()

    words     = args.title.strip().split()
    title_w1  = words[0] if words else "Ranking"
    title_rest = " ".join(words[1:]) if len(words) > 1 else ""

    fo         = _fo()
    music_path = args.music if os.path.exists(args.music) else None
    sep        = "─" * 56

    print(f"\n{'='*56}")
    print(f"  RankZilla Short Builder")
    print(f"  Title    : {args.title}")
    print(f"  Subtitle : {args.subtitle or '(none)'}")
    print(f"  Music    : {music_path or 'none'}")
    print(f"  Output   : {args.output}")
    print(f"{'='*56}")

    with tempfile.TemporaryDirectory() as tmp:

        # ── STEP 1: Download ───────────────────────────────────────────────────
        print(f"\n{sep}\nSTEP 1 / 4  Downloading source video\n{sep}")
        src = os.path.join(tmp, "source.mp4")
        download(args.url, src)

        # ── STEP 2: Analyse & find 5 clip windows ─────────────────────────────
        print(f"\n{sep}\nSTEP 2 / 4  Motion analysis — finding 5 clips\n{sep}")
        print("  Analysing ...")
        ts, sc, fps, duration = analyse_video(src)
        print(f"  Duration: {duration:.1f}s   Samples: {len(ts)}   fps: {fps:.1f}")

        if duration < NUM_CLIPS * MIN_WIN:
            print(f"  WARNING: video is only {duration:.1f}s — "
                  f"clips may be shorter than {MIN_WIN}s")

        windows = find_5_windows(ts, sc, duration)

        print("\n  Ranking (most → least exciting):")
        for rank, w in enumerate(windows, 1):
            print(f"    #{rank}: {w['start']:.1f}s – {w['end']:.1f}s  "
                  f"({w['end'] - w['start']:.1f}s)  score={w['score']:.2f}")

        # ── STEP 3+4: Crop overlays & rebuild with RankZilla branding ─────────
        print(f"\n{sep}\nSTEP 3–4 / 4  Processing segments\n{sep}")

        # Countdown play order: rank #5 (least exciting) → rank #1 (most exciting)
        # windows[0] = most exciting (rank #1)
        # windows[4] = least exciting (rank #5)
        seg_files: list[str]  = []
        seg_durs:  list[float] = []

        for play_pos in range(len(windows)):
            active_rank = len(windows) - play_pos        # 5, 4, 3, 2, 1
            data_idx    = len(windows) - 1 - play_pos    # 4, 3, 2, 1, 0
            w           = windows[data_idx]
            out         = os.path.join(tmp, f"seg_{play_pos:02d}.mp4")

            print(f"\n  [{play_pos + 1}/{len(windows)}] Rank #{active_rank} "
                  f"— {w['start']:.1f}s–{w['end']:.1f}s "
                  f"({w['end'] - w['start']:.1f}s)  score={w['score']:.2f}")

            try:
                dur = process_segment(
                    src,
                    w["start"], w["end"],
                    active_rank,
                    title_w1, title_rest, args.subtitle,
                    fo, out,
                )
            except RuntimeError as exc:
                sys.exit(f"\nFATAL: segment rank#{active_rank} failed:\n{exc}")

            seg_files.append(out)
            seg_durs.append(dur)
            print(f"  Written: {dur:.1f}s → {os.path.basename(out)}")

        # ── Assemble ───────────────────────────────────────────────────────────
        print(f"\n{sep}\nAssembling final video\n{sep}")
        total = sum(seg_durs)
        print(f"  {len(seg_files)} segments, {total:.1f}s total")

        try:
            assemble(seg_files, seg_durs, music_path, args.output, tmp)
        except RuntimeError as exc:
            sys.exit(f"\nFATAL: assembly failed:\n{exc}")

        mb = os.path.getsize(args.output) / (1024 * 1024)
        print(f"\n{'='*56}")
        print(f"  DONE  →  {args.output}  ({mb:.1f} MB)")
        print(f"{'='*56}\n")


if __name__ == "__main__":
    main()
