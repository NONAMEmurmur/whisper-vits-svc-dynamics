import random
import json
import os
from pathlib import Path
from statistics import mean
from concurrent.futures import ThreadPoolExecutor, as_completed

import soundfile as sf

# =========================
# === USER MUST EDIT THESE PATHS AND SETTINGS ===
# （ここを自分の環境に合わせて必ず編集してください / Edit these for your environment）
# =========================

# 入力ディレクトリ（再帰的に *.wav を探す）
INPUT_DIR = Path(r"E:\Music\Whisper_SVC\dataset_raw\NO-NAME")

# 出力する計画JSONのパス
OUTPUT_PLAN_JSON = Path(r"E:\Music\Whisper_SVC\dataset_raw\NO-NAME\NO-NAME_29s_random_plan.json")

MAX_SEC = 29.0          # 1バケットの最大秒数（これを超えるクリップは使わない）

# 何回シャッフルして最良の詰め方を探すか（多いほど良い計画が出やすいが遅い）
SHUFFLE_TRIALS = 20

# 長さ取得を何並列で回すか（None = 自動 = os.cpu_count()）
PROBE_WORKERS = None
# 再現性を持たせたい場合は固定整数。毎回変えたいなら None
RANDOM_SEED = 42
# =========================


def get_wav_duration(path: Path) -> float:
    """
    soundfile で wav の長さ（秒）を取得する。
    """
    info = sf.info(str(path))
    if info.samplerate <= 0:
        raise RuntimeError(f"Invalid samplerate: {path}")
    return info.frames / info.samplerate


def scan_one(path: Path, max_sec: float):
    """
    1ファイル分の長さ取得と判定。
    """
    try:
        dur = get_wav_duration(path)
    except Exception as e:
        return {
            "status": "error",
            "path": str(path),
            "error": repr(e),
        }

    if dur > max_sec:
        return {
            "status": "too_long",
            "path": str(path),
            "duration": dur,
        }

    return {
        "status": "ok",
        "path": str(path),
        "duration": dur,
    }


def scan_wavs_parallel(input_dir: Path, max_sec: float, workers):
    wavs = list(input_dir.rglob("*.wav"))
    if not wavs:
        print("No wav files found.")
        return [], [], []

    print(f"Found wavs: {len(wavs)}")

    if workers is None:
        workers = os.cpu_count() or 4

    print(f"scan workers: {workers}")

    usable = []
    skipped_too_long = []
    skipped_error = []

    completed = 0
    total = len(wavs)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(scan_one, w, max_sec): w for w in wavs}

        for future in as_completed(futures):
            result = future.result()
            completed += 1

            status = result["status"]
            if status == "ok":
                usable.append({
                    "path": result["path"],
                    "duration": result["duration"],
                })
            elif status == "too_long":
                skipped_too_long.append({
                    "path": result["path"],
                    "duration": result["duration"],
                })
            else:
                skipped_error.append({
                    "path": result["path"],
                    "error": result["error"],
                })

            if completed % 1000 == 0 or completed == total:
                print(f"Scanned: {completed}/{total}")

    return usable, skipped_too_long, skipped_error


def make_buckets_greedy_random(clips, max_sec: float, rng: random.Random):
    """
    クリップをシャッフルして、順に max_sec 手前まで詰める。
    最後に残った余りは全部捨てる。
    """
    shuffled = clips[:]
    rng.shuffle(shuffled)

    buckets = []
    current = []
    current_total = 0.0

    for clip in shuffled:
        dur = clip["duration"]

        if current_total + dur <= max_sec:
            current.append(clip)
            current_total += dur
        else:
            if current:
                buckets.append({
                    "total_duration": current_total,
                    "clips": current
                })
            current = [clip]
            current_total = dur

    # 最後に残った余りは、長さやクリップ数に関係なく全部捨てる
    dropped_remainder_bucket = None
    if current:
        dropped_remainder_bucket = {
            "total_duration": current_total,
            "clips": current
        }

    return buckets, dropped_remainder_bucket


def evaluate_buckets(buckets, max_sec: float):
    if not buckets:
        return {
            "bucket_count": 0,
            "avg_sec": 0.0,
            "min_sec": 0.0,
            "max_sec": 0.0,
            "total_sec": 0.0,
            "fill_ratio_avg": 0.0,
        }

    lengths = [b["total_duration"] for b in buckets]
    total_sec = sum(lengths)
    avg_sec = mean(lengths)
    min_sec = min(lengths)
    max_sec_actual = max(lengths)
    fill_ratio_avg = avg_sec / max_sec if max_sec > 0 else 0.0

    return {
        "bucket_count": len(buckets),
        "avg_sec": avg_sec,
        "min_sec": min_sec,
        "max_sec": max_sec_actual,
        "total_sec": total_sec,
        "fill_ratio_avg": fill_ratio_avg,
    }


def choose_best_result(results):
    """
    最良判定の優先順位:
    1. 平均秒数が大きい
    2. バケット数が少ない（= より詰まっている）
    3. 最小秒数が大きい
    """
    return max(
        results,
        key=lambda r: (
            r["metrics"]["avg_sec"],
            -r["metrics"]["bucket_count"],
            r["metrics"]["min_sec"],
        )
    )


def main():
    base_seed = RANDOM_SEED if RANDOM_SEED is not None else random.randrange(1, 10**9)

    usable, skipped_too_long, skipped_error = scan_wavs_parallel(
        INPUT_DIR,
        MAX_SEC,
        PROBE_WORKERS
    )

    print(f"Usable clips         : {len(usable)}")
    print(f"Skipped too long     : {len(skipped_too_long)}")
    print(f"Skipped scan errors  : {len(skipped_error)}")

    if not usable:
        print("No usable wavs.")
        return

    trial_results = []

    print("\n===== SHUFFLE TRIALS START =====")
    for trial_idx in range(1, SHUFFLE_TRIALS + 1):
        trial_seed = base_seed + trial_idx - 1
        rng = random.Random(trial_seed)

        buckets, dropped_remainder_bucket = make_buckets_greedy_random(
            usable,
            MAX_SEC,
            rng
        )

        metrics = evaluate_buckets(buckets, MAX_SEC)

        trial_results.append({
            "trial_index": trial_idx,
            "trial_seed": trial_seed,
            "buckets": buckets,
            "metrics": metrics,
            "dropped_remainder_bucket": dropped_remainder_bucket,
        })

        remainder_drop_str = "yes" if dropped_remainder_bucket is not None else "no"
        remainder_sec = dropped_remainder_bucket["total_duration"] if dropped_remainder_bucket else 0.0
        remainder_clips = len(dropped_remainder_bucket["clips"]) if dropped_remainder_bucket else 0

        print(
            f"[trial {trial_idx:03d}] "
            f"seed={trial_seed} | "
            f"buckets={metrics['bucket_count']} | "
            f"avg_sec={metrics['avg_sec']:.3f} | "
            f"min_sec={metrics['min_sec']:.3f} | "
            f"max_sec={metrics['max_sec']:.3f} | "
            f"remainder_drop={remainder_drop_str} | "
            f"remainder_sec={remainder_sec:.3f} | "
            f"remainder_clips={remainder_clips}"
        )

    best = choose_best_result(trial_results)
    best_metrics = best["metrics"]

    print("\n===== BEST RESULT =====")
    print(f"Chosen trial : {best['trial_index']}")
    print(f"Chosen seed  : {best['trial_seed']}")
    print(f"Bucket count : {best_metrics['bucket_count']}")
    print(f"Average sec  : {best_metrics['avg_sec']:.3f}")
    print(f"Min sec      : {best_metrics['min_sec']:.3f}")
    print(f"Max sec      : {best_metrics['max_sec']:.3f}")
    print(f"Total sec    : {best_metrics['total_sec']:.3f}")
    print(f"Dropped remainder : {'yes' if best['dropped_remainder_bucket'] is not None else 'no'}")
    if best["dropped_remainder_bucket"] is not None:
        print(f"Remainder sec   : {best['dropped_remainder_bucket']['total_duration']:.3f}")
        print(f"Remainder clips : {len(best['dropped_remainder_bucket']['clips'])}")

    plan = {
        "input_dir": str(INPUT_DIR),
        "max_sec": MAX_SEC,
        "shuffle_trials": SHUFFLE_TRIALS,
        "probe_workers": PROBE_WORKERS if PROBE_WORKERS is not None else (os.cpu_count() or 4),
        "base_seed": base_seed,
        "chosen_trial_index": best["trial_index"],
        "chosen_trial_seed": best["trial_seed"],
        "summary": best_metrics,
        "skipped_too_long_count": len(skipped_too_long),
        "skipped_too_long": skipped_too_long,
        "skipped_error_count": len(skipped_error),
        "skipped_error": skipped_error,
        "all_trial_summaries": [
            {
                "trial_index": r["trial_index"],
                "trial_seed": r["trial_seed"],
                "bucket_count": r["metrics"]["bucket_count"],
                "avg_sec": r["metrics"]["avg_sec"],
                "min_sec": r["metrics"]["min_sec"],
                "max_sec": r["metrics"]["max_sec"],
                "total_sec": r["metrics"]["total_sec"],
                "fill_ratio_avg": r["metrics"]["fill_ratio_avg"],
                "dropped_remainder": r["dropped_remainder_bucket"] is not None,
                "remainder_sec": (
                    r["dropped_remainder_bucket"]["total_duration"]
                    if r["dropped_remainder_bucket"] is not None else 0.0
                ),
                "remainder_clip_count": (
                    len(r["dropped_remainder_bucket"]["clips"])
                    if r["dropped_remainder_bucket"] is not None else 0
                ),
            }
            for r in trial_results
        ],
        "dropped_remainder_bucket": best["dropped_remainder_bucket"],
        "buckets": best["buckets"],
    }

    OUTPUT_PLAN_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PLAN_JSON, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)

    print(f"\nSaved plan JSON: {OUTPUT_PLAN_JSON}")


if __name__ == "__main__":
    main()