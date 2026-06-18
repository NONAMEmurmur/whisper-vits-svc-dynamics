import subprocess
import random
import math
import json
from pathlib import Path

# =========================
# === USER MUST EDIT THESE PATHS AND SETTINGS ===
# （ここを自分の環境に合わせて必ず編集してください / Edit these for your environment）
# =========================

# 入力ディレクトリ（再帰的に *.wav を探す）
INPUT_DIR = Path(r"E:\Music\Whisper_SVC\dataset_raw\NO-NAME")

# 出力ディレクトリ（sample_XXXXX.wav と output_plan.json が生成される）
OUTPUT_DIR = Path(r"E:\Music\Whisper_SVC\dataset_raw\NO-NAME\NO-NAME_RMS_disperse_29s")
TMP_DIR = OUTPUT_DIR / "tmp"

MIN_SEC = 2.0
MAX_SEC = 29.0
TARGET_SEC = 29.0

# 1つのNにつき何回まで再計算するか（大きいほど良い計画が出やすいが遅い）
PLAN_RETRIES_PER_N = 200

# 乱数固定したければ整数。毎回変えたいなら None
RANDOM_SEED = 42

# 連結境界に短い無音を挟みたいなら True
INSERT_SILENCE = False
SILENCE_SEC = 0.08
SAMPLE_RATE = 32000
# =========================


def run_cmd(cmd, *, check=True):
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def ffprobe_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path)
    ]
    result = run_cmd(cmd, check=True)
    text = (result.stdout or "").strip()
    if not text:
        raise RuntimeError(f"duration not found for: {path}")
    return float(text)


def ffprobe_stream_info(path: Path):
    """
    wav の音声ストリーム情報を取る。
    サンプリングレート混在や channels 不整合を事前検知する用。
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_type,sample_rate,channels,channel_layout,codec_name",
        "-of", "json",
        str(path)
    ]
    result = run_cmd(cmd, check=True)

    try:
        data = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"ffprobe json parse failed for: {path}\n{e}") from e

    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError(f"no audio stream found for: {path}")

    s = streams[0]
    codec_type = s.get("codec_type")
    sample_rate = s.get("sample_rate")
    channels = s.get("channels")
    channel_layout = s.get("channel_layout")
    codec_name = s.get("codec_name")

    if codec_type != "audio":
        raise RuntimeError(f"first selected stream is not audio: {path}")

    if sample_rate is None:
        raise RuntimeError(f"sample_rate not found for: {path}")

    if channels is None:
        raise RuntimeError(f"channels not found for: {path}")

    return {
        "sample_rate": int(sample_rate),
        "channels": int(channels),
        "channel_layout": channel_layout,
        "codec_name": codec_name,
    }


def ffmpeg_rms_db(path: Path) -> float:
    """
    ffmpeg volumedetect の mean_volume を読む。
    値が大きいほど音が大きい（-20 dB は -35 dB より loud）
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i", str(path),
        "-af", "volumedetect",
        "-f", "null",
        "-"
    ]
    result = run_cmd(cmd, check=False)
    text = (result.stderr or "") + "\n" + (result.stdout or "")

    marker = "mean_volume:"
    for line in text.splitlines():
        if marker in line:
            tail = line.split(marker, 1)[1].strip().replace("dB", "").strip()
            return float(tail)

    raise RuntimeError(f"mean_volume not found for: {path}")


def make_silence_wav(out_path: Path, duration_sec: float):
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "lavfi",
        "-i", f"anullsrc=r={SAMPLE_RATE}:cl=mono",
        "-t", str(duration_sec),
        str(out_path)
    ]
    subprocess.run(cmd, check=True)


def concat_wavs(wav_list, out_path: Path):
    list_file = out_path.with_suffix(".txt")
    with open(list_file, "w", encoding="utf-8") as f:
        for w in wav_list:
            f.write(f"file '{w.as_posix()}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(out_path)
    ]
    subprocess.run(cmd, check=True)
    list_file.unlink(missing_ok=True)


def trim_wav(input_path: Path, out_path: Path, duration: float):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-t", f"{duration:.6f}",
        "-c", "copy",
        str(out_path)
    ]
    subprocess.run(cmd, check=True)


def assign_relative_labels(clips_sorted_desc):
    """
    RMS順位ベースで loud / middle / quiet を付ける。
    固定しきい値は使わない。
    """
    n = len(clips_sorted_desc)
    if n == 0:
        return

    one_third = n // 3
    two_third = (2 * n) // 3

    for i, c in enumerate(clips_sorted_desc):
        if i < one_third:
            c["rel_group"] = "loud"
        elif i < two_third:
            c["rel_group"] = "middle"
        else:
            c["rel_group"] = "quiet"


def total_duration(items):
    return sum(x["duration"] for x in items)


def counts_by_group(items):
    d = {"loud": 0, "middle": 0, "quiet": 0}
    for x in items:
        d[x["rel_group"]] += 1
    return d


def make_candidate_n_list(total_dur):
    """
    実現可能な N の範囲を出し、TARGET_SEC に近い順に候補化する。
    loud_count 制約は撤廃し、総尺だけで範囲を決める。
    """
    lower = math.ceil(total_dur / MAX_SEC)
    upper = math.floor(total_dur / MIN_SEC)

    if lower > upper:
        return []

    target_n = total_dur / TARGET_SEC

    candidates = list(range(lower, upper + 1))
    candidates.sort(key=lambda n: (abs(n - target_n), n))
    return candidates


def choose_bucket_for_clip(buckets, clip):
    """
    clip をどの bucket に入れるか選ぶ。
    - まず MAX_SEC を超えない bucket を候補
    - その中で「入れたあとの空き」が少ない bucket を優先
    - 同じ group が偏っている bucket には少し入れにくくする
    - 少しだけ乱数で崩す
    """
    valid = []
    for idx, b in enumerate(buckets):
        new_total = b["total"] + clip["duration"]
        if new_total <= MAX_SEC:
            remain = MAX_SEC - new_total
            same_group_penalty = b["counts"][clip["rel_group"]] * 0.8
            score = 0.0

            # best-fit 寄り：余りが少ないほど高得点
            score -= remain * 3.0

            # group 偏りを少し嫌う
            score -= same_group_penalty

            # 空の短い bucket を多少優先
            if b["total"] < MIN_SEC:
                score += 0.7

            # 少しだけ崩す
            score += random.uniform(-0.25, 0.25)
            valid.append((score, idx))

    if not valid:
        return None

    valid.sort(reverse=True)
    return valid[0][1]


def interleave_pools(*pools):
    """
    middle / quiet / loud_rest などをラウンドロビンで混ぜる。
    元の「分散して並べる」思想は維持。
    """
    out = []
    max_len = max((len(p) for p in pools), default=0)
    for i in range(max_len):
        for p in pools:
            if i < len(p):
                out.append(p[i])

    # 少しだけ崩すが、完全シャッフルはしない
    for _ in range(min(len(out), 50)):
        if len(out) >= 2:
            a = random.randrange(len(out))
            b = random.randrange(len(out))
            out[a], out[b] = out[b], out[a]
    return out


def try_plan_for_n(all_clips_sorted_desc, n):
    """
    1回分の配分計算。
    成功したら buckets を返し、失敗したら None。
    anchor は「全体RMS上位 N 個」。
    ただし残りの並べ方は middle / quiet / loud_rest を混ぜる思想を維持。
    """
    if len(all_clips_sorted_desc) < n:
        return None

    anchors = all_clips_sorted_desc[:n]
    remaining_pool = all_clips_sorted_desc[n:]

    middle_pool = [c for c in remaining_pool if c["rel_group"] == "middle"]
    quiet_pool = [c for c in remaining_pool if c["rel_group"] == "quiet"]
    loud_rest = [c for c in remaining_pool if c["rel_group"] == "loud"]

    random.shuffle(middle_pool)
    random.shuffle(quiet_pool)
    random.shuffle(loud_rest)

    buckets = []
    for a in anchors:
        buckets.append({
            "clips": [a],
            "total": a["duration"],
            "counts": {
                "loud": 1 if a["rel_group"] == "loud" else 0,
                "middle": 1 if a["rel_group"] == "middle" else 0,
                "quiet": 1 if a["rel_group"] == "quiet" else 0,
            },
        })

    remaining = interleave_pools(middle_pool, quiet_pool, loud_rest)

    for clip in remaining:
        idx = choose_bucket_for_clip(buckets, clip)
        if idx is None:
            return None

        buckets[idx]["clips"].append(clip)
        buckets[idx]["total"] += clip["duration"]
        buckets[idx]["counts"][clip["rel_group"]] += 1

    for b in buckets:
        if not (MIN_SEC <= b["total"] <= MAX_SEC):
            return None

    return buckets


def compute_plan(clips):
    """
    N 候補を試し、各 N ごとに再計算を回す。
    成功するまで ffmpeg は呼ばない。
    """
    total_dur = total_duration(clips)
    candidate_ns = make_candidate_n_list(total_dur)

    if not candidate_ns:
        raise RuntimeError(
            f"No feasible N. total_dur={total_dur:.2f}, "
            f"need ceil(total/MAX)<=N<=floor(total/MIN)"
        )

    print("===== PLAN SEARCH =====")
    print(f"Total duration : {total_dur:.2f}s")
    print(f"Candidate N    : {candidate_ns}")

    for n in candidate_ns:
        print(f"\n--- Try N={n} ---")
        for attempt in range(1, PLAN_RETRIES_PER_N + 1):
            buckets = try_plan_for_n(clips, n)
            if buckets is not None:
                print(f"[SUCCESS] N={n}, attempt={attempt}")
                return n, buckets

            if attempt == 1 or attempt % 20 == 0:
                print(f"[retry] N={n}, attempt={attempt} failed; recomputing...")

    raise RuntimeError("Planning failed for all candidate N values.")


def render_plan(buckets):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    silence_wav = None
    if INSERT_SILENCE:
        silence_wav = TMP_DIR / "_silence.wav"
        if not silence_wav.exists():
            make_silence_wav(silence_wav, SILENCE_SEC)

    plan_dump = []

    for i, b in enumerate(buckets, start=1):
        wav_list = []
        seg_info = []

        for j, c in enumerate(b["clips"], start=1):
            wav_list.append(c["path"])
            seg_info.append({
                "index_in_output": j,
                "path": str(c["path"]),
                "duration": c["duration"],
                "rms_db": c["rms_db"],
                "rel_group": c["rel_group"],
            })
            if INSERT_SILENCE and silence_wav is not None and j != len(b["clips"]):
                wav_list.append(silence_wav)

        tmp_mix = TMP_DIR / f"tmp_{i:05d}.wav"
        out_wav = OUTPUT_DIR / f"sample_{i:05d}.wav"

        concat_wavs(wav_list, tmp_mix)
        final_len = ffprobe_duration(tmp_mix)

        # 計画時に MIN〜MAX に入っているはずだが、
        # concat の都合で微差が出たときだけ軽く trim
        if final_len > MAX_SEC:
            trim_wav(tmp_mix, out_wav, random.uniform(MIN_SEC, MAX_SEC))
            tmp_mix.unlink(missing_ok=True)
        else:
            tmp_mix.rename(out_wav)

        plan_dump.append({
            "output": str(out_wav),
            "planned_total": b["total"],
            "actual_total_before_trim": final_len,
            "counts": b["counts"],
            "clips": seg_info,
        })

        if i % 20 == 0:
            print(
                f"[render {i:05d}] "
                f"dur≈{b['total']:.2f}s "
                f"l/m/q={b['counts']['loud']}/{b['counts']['middle']}/{b['counts']['quiet']}"
            )

    plan_json = OUTPUT_DIR / "output_plan.json"
    with open(plan_json, "w", encoding="utf-8") as f:
        json.dump(plan_dump, f, ensure_ascii=False, indent=2)

    print(f"Saved: {plan_json}")


def validate_input_wavs(clips):
    """
    事前に罠を検出して明示エラーにする。
    - 単体29秒超え
    - サンプリングレート混在
    - channels 混在
    """
    over_max = [c for c in clips if c["duration"] > MAX_SEC + 1e-6]
    if over_max:
        sample = "\n".join(
            f"  {c['path']} | duration={c['duration']:.6f}s"
            for c in over_max[:20]
        )
        more = ""
        if len(over_max) > 20:
            more = f"\n  ... and {len(over_max) - 20} more"
        raise RuntimeError(
            "Found wav(s) longer than MAX_SEC. "
            "This script does not auto-split a single overlong wav.\n"
            f"MAX_SEC={MAX_SEC}\n"
            f"{sample}{more}"
        )

    sample_rates = sorted({c["sample_rate"] for c in clips})
    if len(sample_rates) > 1:
        offenders = {}
        for c in clips:
            offenders.setdefault(c["sample_rate"], []).append(c["path"])

        lines = []
        for sr, paths in offenders.items():
            lines.append(f"  sample_rate={sr}")
            for p in paths[:10]:
                lines.append(f"    {p}")
            if len(paths) > 10:
                lines.append(f"    ... and {len(paths) - 10} more")

        raise RuntimeError(
            "Mixed sample rates detected. "
            "This script uses concat demuxer with stream-copy, so mixed SR is a trap.\n"
            + "\n".join(lines)
        )

    channels_set = sorted({c["channels"] for c in clips})
    if len(channels_set) > 1:
        offenders = {}
        for c in clips:
            offenders.setdefault(c["channels"], []).append(c["path"])

        lines = []
        for ch, paths in offenders.items():
            lines.append(f"  channels={ch}")
            for p in paths[:10]:
                lines.append(f"    {p}")
            if len(paths) > 10:
                lines.append(f"    ... and {len(paths) - 10} more")

        raise RuntimeError(
            "Mixed channel counts detected. "
            "This also tends to break concat/copy workflows in annoying ways.\n"
            + "\n".join(lines)
        )


def main():
    random.seed(RANDOM_SEED)

    wavs = list(INPUT_DIR.rglob("*.wav"))
    if not wavs:
        print("No wav files found.")
        return

    print(f"Found wavs: {len(wavs)}")

    clips = []
    for i, w in enumerate(wavs, start=1):
        stream_info = ffprobe_stream_info(w)
        dur = ffprobe_duration(w)
        rms = ffmpeg_rms_db(w)

        clips.append({
            "path": w,
            "duration": dur,
            "rms_db": rms,
            "sample_rate": stream_info["sample_rate"],
            "channels": stream_info["channels"],
            "channel_layout": stream_info["channel_layout"],
            "codec_name": stream_info["codec_name"],
        })

        if i % 200 == 0:
            print(f"Scanned duration+RMS: {i}/{len(wavs)}")

    validate_input_wavs(clips)

    # RMS 相対順位だけで group を付ける
    clips_sorted_desc = sorted(clips, key=lambda x: x["rms_db"], reverse=True)
    assign_relative_labels(clips_sorted_desc)

    group_counts = {"loud": 0, "middle": 0, "quiet": 0}
    for c in clips_sorted_desc:
        group_counts[c["rel_group"]] += 1

    print("===== RELATIVE RMS GROUPS =====")
    print(f"loud   : {group_counts['loud']}")
    print(f"middle : {group_counts['middle']}")
    print(f"quiet  : {group_counts['quiet']}")

    sample_rate = clips_sorted_desc[0]["sample_rate"] if clips_sorted_desc else None
    channels = clips_sorted_desc[0]["channels"] if clips_sorted_desc else None
    print("===== INPUT AUDIO FORMAT =====")
    print(f"sample_rate : {sample_rate}")
    print(f"channels    : {channels}")

    # まず計画だけ
    n, buckets = compute_plan(clips_sorted_desc)

    print("\n===== PLAN SUMMARY =====")
    print(f"Chosen N: {n}")
    for i, b in enumerate(buckets[:10]):
        print(
            f"[plan {i:03d}] "
            f"dur≈{b['total']:.2f}s "
            f"l/m/q={b['counts']['loud']}/{b['counts']['middle']}/{b['counts']['quiet']}"
        )

    # 計画成功後にだけ ffmpeg
    print("\n===== RENDER START =====")
    render_plan(buckets)

    print("===== DONE =====")


if __name__ == "__main__":
    main()
