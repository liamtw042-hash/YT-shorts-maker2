#!/usr/bin/env python3
"""
make_short.py — RankZilla-style ranked YouTube Short generator.

Reads 5 YouTube URLs from clips.txt, ranks them by excitement score
(motion analysis), then assembles a countdown reveal Short:
  rank #5 (least exciting) first → rank #1 (most exciting) last.

Usage:
    python make_short.py --title "Ranking Funniest" --subtitle "Memes 2026"

clips.txt format — one URL per line, exactly 5 lines:
    https://youtube.com/shorts/xxx
    https://youtube.com/shorts/xxx
    ...

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


# ── Layout constants ───────────────────────────────────────────────────────────

TARGET_W  = 1080
TARGET_H  = 1920
FPS       = 30
BANNER_H  = 120          # title strip height at top
NUM_CLIPS = 5
MAX_RETRY = 3

# Vertical center (px) for each rank label in the left sidebar.
# Usable area below banner: y=120 to y=1920 (1800 px).
# 5 slots, 200 px apart, centred in the usable area.
_CTR = 1020              # midpoint of usable area
RANK_Y = {r: _CTR + (r - 3) * 200 for r in range(1, 6)}
# → rank1:620  rank2:820  rank3:1020  rank4:1220  rank5:1420

MIN_WIN = 4.0            # minimum clip window (seconds)
MAX_WIN = 8.0            # maximum clip window (seconds)

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
    """fontfile option string for ffmpeg drawtext, or '' if not found."""
    fp = _find_font()
    if not fp:
        return ""
    return f":fontfile='{fp.replace(chr(92), '/').replace(':', chr(92) + ':')}'"


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
    """
    Run a subprocess command.  On failure, print full stderr and raise.
    On success, stay silent (ffmpeg is very chatty).
    """
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tag = f" [{label}]" if label else ""
        raise RuntimeError(
            f"Command failed{tag} (exit {result.returncode})\n"
            f"CMD: {' '.join(str(x) for x in cmd[:8])} ...\n"
            f"STDERR (last 4000 chars):\n{result.stderr[-4000:]}"
        )


def _verify(path: str, label: str = "") -> None:
    """Raise if path does not exist or is empty."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise RuntimeError(
            f"Expected output missing or empty{' [' + label + ']' if label else ''}: {path}"
        )


# ── Download ───────────────────────────────────────────────────────────────────

def download_clip(url: str, dest: str) -> None:
    """
    Download with up to MAX_RETRY attempts.
    Verifies the file exists and is non-empty.
    Exits the whole program on permanent failure — never silently skips.
    """
    for attempt in range(1, MAX_RETRY + 1):
        print(f"  [{attempt}/{MAX_RETRY}] Downloading: {url}")
        try:
            subprocess.run(
                [
                    sys.executable, "-m", "yt_dlp",
                    "--no-playlist",
                    "--merge-output-format", "mp4",
                    "--format",
                    "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best",
                    "--retries", "10",
                    "--fragment-retries", "10",
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
                print(f"  Waiting {wait}s before retry...")
                time.sleep(wait)
                continue
            sys.exit(
                f"\nERROR: Could not download after {MAX_RETRY} attempts.\n"
                f"URL: {url}\nLast error: {exc}"
            )

        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            kb = os.path.getsize(dest) // 1024
            print(f"  OK ({kb} KB): {os.path.basename(dest)}")
            return

        print(f"  File missing or empty after attempt {attempt}.")
        if attempt == MAX_RETRY:
            sys.exit(f"\nERROR: Download produced empty file after {MAX_RETRY} attempts.\nURL: {url}")
        time.sleep(4 * attempt)


# ── Motion analysis ────────────────────────────────────────────────────────────

def analyse_video(path: str) -> tuple[list, list, float, float]:
    """
    Sample frame-diff motion at ~4 fps using cv2.
    Returns (timestamps, scores, fps, duration).
    """
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps if total > 0 else 0.0
    step = max(1, int(fps / 4))

    timestamps, scores, prev_gray, idx = [], [], None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            sm = cv2.resize(frame, (320, 240))
            gray = cv2.GaussianBlur(
                cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY), (21, 21), 0
            )
            if prev_gray is not None:
                timestamps.append(idx / fps)
                scores.append(float(np.mean(cv2.absdiff(prev_gray, gray))))
            prev_gray = gray
        idx += 1

    cap.release()
    return timestamps, scores, fps, duration


def excitement_score(scores: list) -> float:
    """80th-percentile motion as clip excitement metric."""
    return float(np.percentile(scores, 80)) if scores else 0.0


def best_window(timestamps: list, scores: list, duration: float) -> tuple[float, float]:
    """
    Find the highest-motion natural action window (MIN_WIN–MAX_WIN seconds).
    Walks outward from the peak frame until motion drops below 30% of peak,
    then clamps duration to [MIN_WIN, MAX_WIN].
    """
    if duration <= MAX_WIN:
        return 0.0, duration
    if not scores:
        return 0.0, min(MAX_WIN, duration)

    ts = np.array(timestamps)
    sc = np.array(scores)
    pi = int(np.argmax(sc))
    pt = float(ts[pi])
    thresh = float(sc[pi]) * 0.30
    walk = int(MAX_WIN * 4)   # samples at ~4 fps

    # Walk left from peak
    s = pt
    for i in range(pi - 1, max(0, pi - walk), -1):
        if sc[i] < thresh:
            s = float(ts[i + 1])
            break
        s = float(ts[i])

    # Walk right from peak
    e = pt
    for i in range(pi + 1, min(len(ts), pi + walk)):
        if sc[i] < thresh:
            e = float(ts[i - 1])
            break
        e = float(ts[i])

    # Enforce min/max duration
    dur = e - s
    if dur < MIN_WIN:
        mid = (s + e) / 2.0
        s, e = mid - MIN_WIN / 2, mid + MIN_WIN / 2
    elif dur > MAX_WIN:
        mid = (s + e) / 2.0
        s, e = mid - MAX_WIN / 2, mid + MAX_WIN / 2

    return max(0.0, s), min(duration, e)


# ── Video filter builders ──────────────────────────────────────────────────────

def _banner_filters(w1: str, rest: str, subtitle: str, fo: str) -> list:
    """
    Title banner overlay filters.
    w1 = first word (white), rest = remaining title words (red),
    subtitle = yellow line below.
    """
    x_rest = 40 + int(len(w1) * 54 * 0.60) + 10  # approximate char width

    parts = [
        f"drawbox=x=0:y=0:w={TARGET_W}:h={BANNER_H}:color=black:t=fill",
        f"drawtext=text='{_esc(w1)}'{fo}"
        f":fontsize=54:fontcolor=white:borderw=3:bordercolor=black:x=40:y=18",
    ]
    if rest:
        parts.append(
            f"drawtext=text='{_esc(rest)}'{fo}"
            f":fontsize=54:fontcolor='0xFF3333':borderw=3:bordercolor=black"
            f":x={x_rest}:y=18"
        )
    if subtitle:
        parts.append(
            f"drawtext=text='{_esc(subtitle)}'{fo}"
            f":fontsize=32:fontcolor=yellow:borderw=2:bordercolor=black:x=40:y=78"
        )
    return parts


def _rank_filters(active: int, fo: str) -> list:
    """
    Left-sidebar rank number filters.
    active rank: 85 px bold yellow/white with thick black stroke.
    others: 48 px grey semi-transparent.
    """
    parts = []
    for r in range(1, NUM_CLIPS + 1):
        cy = RANK_Y[r]
        if r == active:
            fs, color, bw, bc = 85, "yellow", 7, "black"
            y = cy - 42
        else:
            fs, color, bw, bc = 48, "'0xBBBBBB@0.60'", 2, "'0x00000066'"
            y = cy - 24
        parts.append(
            f"drawtext=text='{r}\.'{fo}"
            f":fontsize={fs}:fontcolor={color}"
            f":borderw={bw}:bordercolor={bc}:x=30:y={y}"
        )
    return parts


def build_vf(active_rank: int, w1: str, rest: str, subtitle: str, fo: str) -> str:
    """
    Complete -vf filter chain for one clip segment:
      letterbox → 1.05x zoom → title banner → rank sidebar
    """
    chain = [
        # Letterbox to 1080x1920 preserving aspect ratio
        f"scale=w={TARGET_W}:h={TARGET_H}:force_original_aspect_ratio=decrease",
        f"pad={TARGET_W}:{TARGET_H}:(ow-iw)/2:(oh-ih)/2:black",
        # 1.05× zoom to hide corner watermarks/text
        f"scale=iw*1.05:ih*1.05",
        f"crop={TARGET_W}:{TARGET_H}",
        # Force 30 fps
        f"fps={FPS}",
    ]
    chain += _banner_filters(w1, rest, subtitle, fo)
    chain += _rank_filters(active_rank, fo)
    return ",".join(chain)


# ── Segment processing ─────────────────────────────────────────────────────────

def process_segment(
    source: str,
    t_start: float, t_end: float,
    active_rank: int,
    w1: str, rest: str, subtitle: str,
    fo: str,
    out: str,
) -> float:
    """
    Render one clip window with overlays burned in.
    Returns actual clip duration (seconds).
    Raises RuntimeError (with full stderr) on ffmpeg failure.
    """
    dur = round(t_end - t_start, 3)
    vf = build_vf(active_rank, w1, rest, subtitle, fo)
    af = "loudnorm=I=-16:TP=-1.5:LRA=11"

    base = [
        "ffmpeg", "-y",
        "-ss", str(t_start), "-i", source,
        "-t", str(dur),
        "-vf", vf,
    ]

    if _has_audio(source):
        cmd = base + [
            "-af", af,
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

    _run(cmd, label=f"segment rank#{active_rank}")
    _verify(out, f"segment rank#{active_rank}")
    return dur


# ── Assembly ───────────────────────────────────────────────────────────────────

def assemble(
    seg_files: list,
    seg_durs: list,
    music_path: str | None,
    output: str,
    tmp: str,
) -> None:
    """Concatenate segments; optionally mix background music at 10% volume."""
    total_dur = sum(seg_durs)

    concat_txt = os.path.join(tmp, "concat.txt")
    with open(concat_txt, "w") as f:
        for sf, dur in zip(seg_files, seg_durs):
            f.write(f"file '{sf}'\nduration {dur}\n")

    concat_raw = os.path.join(tmp, "concat_raw.mp4")
    print("  Concatenating segments...")
    _run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_txt, "-c", "copy", concat_raw,
    ], label="concat")
    _verify(concat_raw, "concat")

    if not music_path:
        shutil.copy2(concat_raw, output)
        return

    # Pre-loop music to total duration
    looped = os.path.join(tmp, "music_loop.wav")
    print(f"  Looping music to {total_dur:.1f}s...")
    try:
        _run([
            "ffmpeg", "-y",
            "-stream_loop", "-1", "-i", music_path,
            "-t", str(total_dur),
            "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "2",
            looped,
        ], label="music loop")
    except RuntimeError as exc:
        print(f"  WARNING: music loop failed, skipping music.\n  {exc}")
        shutil.copy2(concat_raw, output)
        return

    # Mix original audio + music at 10%
    print("  Mixing music and exporting final video...")
    _run([
        "ffmpeg", "-y",
        "-i", concat_raw,
        "-i", looped,
        "-filter_complex",
        "[1:a]volume=0.10[bg];"
        "[0:a][bg]amix=inputs=2:duration=first:dropout_transition=0[aout]",
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        output,
    ], label="final mix")
    _verify(output, "final output")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="RankZilla-style YouTube Short generator.")
    parser.add_argument("--title",    default="Ranking Top Moments",
        help="Title text. First word shown in white, rest in red.")
    parser.add_argument("--subtitle", default="",
        help="Subtitle shown in yellow below the title.")
    parser.add_argument("--clips",   default="clips.txt", metavar="FILE")
    parser.add_argument("--music",   default="music.mp3",  metavar="FILE")
    parser.add_argument("--output",  default="output.mp4", metavar="FILE")
    args = parser.parse_args()

    # Parse title into first word (white) + remainder (red)
    words = args.title.strip().split()
    title_w1   = words[0] if words else "Ranking"
    title_rest = " ".join(words[1:]) if len(words) > 1 else ""

    # Load clips.txt
    if not os.path.exists(args.clips):
        sys.exit(
            f"ERROR: {args.clips} not found.\n"
            "Create it with one YouTube URL per line (5 lines)."
        )
    with open(args.clips) as f:
        urls = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if not urls:
        sys.exit(f"ERROR: No URLs found in {args.clips}")
    if len(urls) < NUM_CLIPS:
        print(f"WARNING: Expected {NUM_CLIPS} URLs, found {len(urls)}. Proceeding with {len(urls)}.")
    urls = urls[:NUM_CLIPS]

    fo = _fo()
    music_path = args.music if os.path.exists(args.music) else None
    if music_path:
        print(f"Music: {args.music} at 10% volume")
    else:
        print("No music.mp3 — skipping background music")

    # ─────────────────────────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmp:

        # ── STEP 1: Download ───────────────────────────────────────────────────
        sep = "─" * 52
        print(f"\n{sep}\nSTEP 1 / 4  Downloading {len(urls)} clips\n{sep}")
        src_paths: list[str] = []
        for i, url in enumerate(urls):
            print(f"\n[{i+1}/{len(urls)}]")
            dest = os.path.join(tmp, f"src_{i}.mp4")
            download_clip(url, dest)
            src_paths.append(dest)

        # ── STEP 2: Analyse & rank ─────────────────────────────────────────────
        print(f"\n{sep}\nSTEP 2 / 4  Analysing motion & ranking clips\n{sep}")
        clips: list[dict] = []
        for i, (path, url) in enumerate(zip(src_paths, urls)):
            print(f"  Analysing clip {i+1}/{len(src_paths)} ...")
            ts, sc, fps, dur = analyse_video(path)
            exc = excitement_score(sc)
            ws, we = best_window(ts, sc, dur)
            clips.append(dict(path=path, url=url, score=exc,
                               ws=ws, we=we, dur=dur))
            print(f"    duration={dur:.1f}s  excitement={exc:.2f}"
                  f"  window={ws:.1f}–{we:.1f}s ({we-ws:.1f}s)")

        # Sort descending by excitement → clips[0] = most exciting = rank #1
        clips.sort(key=lambda c: c["score"], reverse=True)

        print("\n  Final ranking:")
        for rank, c in enumerate(clips, 1):
            print(f"    #{rank}  score={c['score']:.2f}  {c['url']}")

        # ── STEP 3: Process segments ───────────────────────────────────────────
        print(f"\n{sep}\nSTEP 3 / 4  Processing clip segments\n{sep}")

        # Play order: rank #5 first (index 4), rank #1 last (index 0)
        seg_files: list[str] = []
        seg_durs:  list[float] = []

        for play_pos, data_idx in enumerate(range(len(clips) - 1, -1, -1)):
            active_rank = len(clips) - play_pos   # 5, 4, 3, 2, 1
            c = clips[data_idx]
            out = os.path.join(tmp, f"seg_{play_pos:02d}.mp4")

            print(f"\n  Segment {play_pos+1}/{len(clips)}: Rank #{active_rank}"
                  f"  (excitement={c['score']:.2f})")
            print(f"    window {c['ws']:.1f}s–{c['we']:.1f}s"
                  f"  ({c['we']-c['ws']:.1f}s)")

            try:
                dur = process_segment(
                    c["path"], c["ws"], c["we"],
                    active_rank,
                    title_w1, title_rest, args.subtitle,
                    fo, out,
                )
            except RuntimeError as exc:
                sys.exit(f"\nERROR processing rank #{active_rank}:\n{exc}")

            seg_files.append(out)
            seg_durs.append(dur)
            print(f"    Written: {dur:.1f}s -> {os.path.basename(out)}")

        # ── STEP 4: Assemble ───────────────────────────────────────────────────
        print(f"\n{sep}\nSTEP 4 / 4  Assembling final video\n{sep}")
        total = sum(seg_durs)
        print(f"  {len(seg_files)} segments, total {total:.1f}s")

        try:
            assemble(seg_files, seg_durs, music_path, args.output, tmp)
        except RuntimeError as exc:
            sys.exit(f"\nERROR assembling video:\n{exc}")

        size_mb = os.path.getsize(args.output) / (1024 * 1024)
        print(f"\n{'='*52}")
        print(f"  Done!  {args.output}  ({size_mb:.1f} MB)")
        print(f"{'='*52}")


if __name__ == "__main__":
    main()
