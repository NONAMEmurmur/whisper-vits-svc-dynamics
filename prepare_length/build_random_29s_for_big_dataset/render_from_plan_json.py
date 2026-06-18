import subprocess
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================
# === USER MUST EDIT THESE PATHS AND SETTINGS ===
# （ここを自分の環境に合わせて必ず編集してください / Edit these for your environment）
# =========================

# 入力する計画JSON（build_*_plan_*.py で生成したもの）
INPUT_PLAN_JSON = Path(r"E:\Music\Whisper_SVC\dataset_raw\NO-NAME\NO-NAME_29s_random_plan.json")

# 出力ディレクトリ（sample_XXXXX.wav が生成される）
OUTPUT_DIR = Path(r"E:\Music\Whisper_SVC\dataset_raw\NO-NAME\NO-NAME_29s_random")

# 同時に何本 render するか
# None なら CPU 論理コア数を使う
# ただし ffmpeg はディスクI/Oも使うので、まずは 4〜8 くらい推奨
RENDER_WORKERS = 4

# 実際に ffmpeg を動かす前に dry run したいなら True
DRY_RUN = False
# =========================


def ffprobe_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def concat_wavs(wav_list, out_path: Path):
    list_file = out_path.with_suffix(".txt")

    with open(list_file, "w", encoding="utf-8") as f:
        for w in wav_list:
            # ffmpeg concat demuxer 用
            f.write(f"file '{Path(w).as_posix()}'\n")

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


def render_one(index: int, bucket: dict, output_dir: Path, dry_run: bool = False):
    out_wav = output_dir / f"sample_{index:05d}.wav"
    wav_list = [clip["path"] for clip in bucket["clips"]]

    if dry_run:
        return {
            "status": "dry_run",
            "index": index,
            "output": str(out_wav),
            "planned_total_duration": bucket["total_duration"],
            "actual_duration": None,
            "clip_count": len(bucket["clips"]),
            "clips": bucket["clips"],
        }

    concat_wavs(wav_list, out_wav)

    try:
        actual_dur = ffprobe_duration(out_wav)
    except Exception:
        actual_dur = None

    return {
        "status": "ok",
        "index": index,
        "output": str(out_wav),
        "planned_total_duration": bucket["total_duration"],
        "actual_duration": actual_dur,
        "clip_count": len(bucket["clips"]),
        "clips": bucket["clips"],
    }


def main():
    if not INPUT_PLAN_JSON.exists():
        print(f"Plan JSON not found: {INPUT_PLAN_JSON}")
        return

    with open(INPUT_PLAN_JSON, "r", encoding="utf-8") as f:
        plan = json.load(f)

    buckets = plan.get("buckets", [])
    if not buckets:
        print("No buckets in plan JSON.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    workers = RENDER_WORKERS
    if workers is None:
        workers = os.cpu_count() or 4

    print(f"Bucket count   : {len(buckets)}")
    print(f"Render workers : {workers}")
    print(f"Dry run        : {DRY_RUN}")

    rendered = []
    failed = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(render_one, i, bucket, OUTPUT_DIR, DRY_RUN): i
            for i, bucket in enumerate(buckets)
        }

        completed = 0
        total = len(futures)

        for future in as_completed(futures):
            idx = futures[future]
            completed += 1

            try:
                result = future.result()
                rendered.append(result)

                planned = result["planned_total_duration"]
                actual = result["actual_duration"]
                actual_str = f"{actual:.3f}s" if actual is not None else "N/A"

                print(
                    f"[done {completed:05d}/{total:05d}] "
                    f"sample_{idx:05d}.wav | "
                    f"planned={planned:.3f}s | "
                    f"actual={actual_str} | "
                    f"clips={result['clip_count']}"
                )

            except Exception as e:
                failed.append({
                    "index": idx,
                    "error": repr(e),
                })
                print(
                    f"[FAIL {completed:05d}/{total:05d}] "
                    f"sample_{idx:05d}.wav | {repr(e)}"
                )

    # index順に並べ直して保存
    rendered.sort(key=lambda x: x["index"])
    failed.sort(key=lambda x: x["index"])

    render_log_json = OUTPUT_DIR / "render_log.json"
    with open(render_log_json, "w", encoding="utf-8") as f:
        json.dump({
            "input_plan_json": str(INPUT_PLAN_JSON),
            "output_dir": str(OUTPUT_DIR),
            "render_workers": workers,
            "dry_run": DRY_RUN,
            "bucket_count": len(buckets),
            "rendered_count": len(rendered),
            "failed_count": len(failed),
            "failed": failed,
            "rendered": rendered,
        }, f, ensure_ascii=False, indent=2)

    print(f"\nSaved render log: {render_log_json}")
    print(f"Rendered: {len(rendered)}")
    print(f"Failed  : {len(failed)}")
    print("DONE")


if __name__ == "__main__":
    main()