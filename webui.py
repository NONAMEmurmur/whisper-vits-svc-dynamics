"""
Acoustic Dynamics 推論 UI

元リポジトリ付属の app.py（ワンショット推論）は使わない。
このフォーク推奨の二段階推論をそのまま画面にした Gradio UI。

  Stage 1  pitch/inference.py   F0 + periodicity を作業フォルダへ保存
  Stage 2  選んだ学習フォルダのモデル / 話者で変換

起動:
  python webui.py
  python webui.py --port 7860 --listen
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SETTINGS_PATH = ROOT / "webui_settings.json"
PREVIEW_DIR = ROOT / "outputs" / "_gradio_preview"
SKIP_CKPT_PARTS = {
    "_step_ckpt_trash",
    "_pruned_checkpoints",
    "corrupted",
    "__pycache__",
}
SKIP_REPO_MARKERS = ("パッチ", "学習データ", "データセット空")


# ---------------------------------------------------------------------------
# device / cache
# ---------------------------------------------------------------------------

def detect_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


DEFAULT_DEVICE = detect_device()

_model_cache = {
    "model": None,
    "path": None,
    "config": None,
    "hp": None,
    "device": None,
}


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------

def load_settings() -> dict:
    if SETTINGS_PATH.is_file():
        try:
            return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "work_dir": "",
        "last_repo": str(ROOT),
        "extra_repos": [],
    }


def save_settings(**kwargs) -> dict:
    data = load_settings()
    data.update(kwargs)
    SETTINGS_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return data


# ---------------------------------------------------------------------------
# path helpers
# ---------------------------------------------------------------------------

def as_path(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return as_path(value[0])
    if isinstance(value, dict):
        for key in ("path", "name"):
            if value.get(key):
                return str(value[key])
        return None
    if hasattr(value, "name") and not isinstance(value, (str, Path)):
        return str(value.name)
    text = str(value).strip()
    return text or None


def is_temp_path(path: str | Path) -> bool:
    text = str(path)
    markers = (
        "gradio",
        "AppData\\Local\\Temp",
        "AppData/Local/Temp",
        "\\Temp\\",
        "/tmp/",
    )
    return any(marker in text for marker in markers)


def parse_float(value, name: str) -> float:
    text = str(value).strip().replace(",", "")
    if not text:
        raise ValueError(f"{name} が空です。")
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"{name} が数値ではありません: {value}") from exc


def parse_int(value, name: str) -> int:
    return int(parse_float(value, name))


def sanitize_filename(name: str) -> str:
    bad = '<>:"/\\|?*'
    cleaned = "".join("_" if ch in bad else ch for ch in str(name))
    return cleaned.strip(" .") or "out"


def speaker_label(path: str | Path) -> str:
    name = Path(path).name
    if name.endswith(".spk.npy"):
        return name[: -len(".spk.npy")]
    return Path(path).stem


def unique_path(path: Path, overwrite: bool) -> Path:
    if overwrite or not path.exists():
        return path
    for i in range(2, 1000):
        candidate = path.with_name(f"{path.stem}_{i}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"空きファイル名を作れませんでした: {path}")


def require_work_dir(work_dir) -> Path:
    text = as_path(work_dir)
    if not text:
        raise FileNotFoundError("先に作業フォルダを指定して「適用」してください。")
    folder = Path(text)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def uv_filename_token(uv_text, uv_value: float) -> str:
    raw = str(uv_text).strip().replace(" ", "")
    try:
        parsed = float(raw)
        if abs(parsed - uv_value) <= max(abs(uv_value) * 1e-9, 1e-20):
            return sanitize_filename(raw)
    except Exception:
        pass
    return sanitize_filename(f"{uv_value:.15g}")


def default_csv_name(
    wav_path: str,
    method: str,
    uv_text,
    uv: float,
    post: bool,
    harmonic: bool,
    fill_gaps: bool = False,
    max_gap_ms: float = 3.0,
) -> str:
    extras = ""
    if method == "rmvpe" and harmonic:
        extras += "_h"
    if post:
        extras += "_post"
    if fill_gaps:
        extras += f"_gap{float(max_gap_ms):g}"
    return f"{Path(wav_path).stem}_{method}{extras}_uv{uv_filename_token(uv_text, uv)}.csv"


def format_weight(weight: float) -> str:
    text = f"{float(weight):.4g}"
    return text


def parse_weight(value) -> float:
    text = str(value).strip().replace(",", ".")
    if not text:
        return 0.0
    return float(text)


def serve_for_gradio(path: str | Path | None) -> str | None:
    """Copy a work-folder file into the repo so Gradio can play it."""
    if not path:
        return None
    src = Path(path)
    if not src.is_file():
        return None
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    dest = PREVIEW_DIR / src.name
    if dest.resolve() != src.resolve():
        shutil.copy2(src, dest)
    return str(dest)


def pick_directory(initial: str | None = None) -> str | None:
    """エクスプローラー型フォルダ選択。軽量な別プロセスで出す。"""
    import subprocess

    cmd = [sys.executable, str(ROOT / "folder_picker.py")]
    start = as_path(initial)
    if start:
        cmd += ["--initial", start]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(ROOT),
    )
    chosen = (proc.stdout or "").strip()
    if proc.returncode != 0 and not chosen:
        err = (proc.stderr or "").strip()
        if err:
            print(err)
        return None
    return chosen or None


def collect_allowed_paths() -> list[str]:
    paths = [str(ROOT), str(ROOT / "outputs"), str(PREVIEW_DIR)]
    settings = load_settings()
    work = as_path(settings.get("work_dir"))
    if work:
        paths.append(str(Path(work).resolve()))
        parent = str(Path(work).resolve().parent)
        paths.append(parent)
    for extra in (r"E:\Music", r"E:\music", r"E:\AI"):
        if Path(extra).exists():
            paths.append(extra)
    for repo in settings.get("extra_repos") or []:
        if repo:
            paths.append(str(Path(repo).resolve()))
    seen: set[str] = set()
    out: list[str] = []
    for item in paths:
        key = os.path.normcase(os.path.normpath(item))
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def default_wav_name(csv_path: str | Path, speaker: str) -> str:
    return f"{Path(csv_path).stem}_{sanitize_filename(speaker)}.wav"


def prd_beside(csv_path: str | Path) -> Path:
    csv_path = Path(csv_path)
    return csv_path.with_name(csv_path.stem + ".prd.npy")


def resolve_csv_and_prd(csv_value, work_dir) -> tuple[Path, Path]:
    raw = as_path(csv_value)
    if not raw:
        raise FileNotFoundError("ピッチCSVを選んでください。")

    csv_path = Path(raw)
    folder = None
    try:
        folder = require_work_dir(work_dir)
    except Exception:
        folder = None

    if (not csv_path.is_file() or is_temp_path(csv_path)) and folder is not None:
        local = folder / csv_path.name
        if local.is_file():
            csv_path = local

    if not csv_path.is_file():
        raise FileNotFoundError(f"ピッチCSVが見つかりません: {raw}")

    prd = prd_beside(csv_path)
    if not prd.is_file() and folder is not None:
        alt = folder / prd.name
        if alt.is_file():
            prd = alt
    if not prd.is_file():
        raise FileNotFoundError(
            f".prd.npy がありません: {prd}\n"
            "CSV を Gradio にドロップすると一時フォルダへコピーされ、prd が置いていかれます。\n"
            "作業フォルダのドロップダウンから選ぶか、Stage 1 で作り直してください。"
        )
    return csv_path, prd


def materialize_wav(uploaded, work_dir: Path) -> str:
    src = as_path(uploaded)
    if not src or not Path(src).is_file():
        raise FileNotFoundError("入力WAVを選んでください。")
    src_path = Path(src)
    dest = work_dir / src_path.name
    if src_path.resolve() == dest.resolve():
        return str(dest)
    if (not dest.is_file()) or dest.stat().st_size != src_path.stat().st_size:
        shutil.copy2(src_path, dest)
    return str(dest)


def cache_feature_path(work_dir: Path, wav_path: str, suffix: str) -> Path:
    import hashlib

    digest = hashlib.md5(os.path.abspath(wav_path).encode("utf-8")).hexdigest()[:8]
    cache = work_dir / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    return cache / f"{Path(wav_path).stem}_{digest}{suffix}"


# ---------------------------------------------------------------------------
# repo / model / speaker scan
# ---------------------------------------------------------------------------

def repo_short_name(path: Path) -> str:
    name = path.name
    for prefix in ("whisper-vits-svc-", "whisper-vits-svc_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    if path.resolve() == ROOT.resolve():
        return f"{name}（このUI）"
    return name


def is_svc_repo(path: Path) -> bool:
    return path.is_dir() and (path / "chkpt").is_dir() and (
        (path / "svc_inference.py").is_file() or (path / "data_svc" / "singer").is_dir()
    )


def should_skip_repo(path: Path) -> bool:
    name = path.name
    return any(marker in name for marker in SKIP_REPO_MARKERS)


def scan_repos(extra_repos: list[str] | None = None) -> list[tuple[str, str]]:
    found: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path):
        try:
            resolved = path.resolve()
        except Exception:
            return
        if resolved in seen or not is_svc_repo(resolved):
            return
        seen.add(resolved)
        found.append(resolved)

    add(ROOT)
    parent = ROOT.parent
    if parent.is_dir():
        for child in sorted(parent.iterdir()):
            if child.is_dir() and not should_skip_repo(child):
                add(child)

    for item in extra_repos or load_settings().get("extra_repos") or []:
        if item:
            add(Path(item))

    return [(repo_short_name(path), str(path)) for path in found]


def scan_repo_models(repo: str | Path) -> list[tuple[str, str]]:
    chkpt = Path(repo) / "chkpt"
    items: list[tuple[str, str]] = []
    if not chkpt.is_dir():
        return items
    for path in sorted(chkpt.rglob("*.pt")):
        if any(part in SKIP_CKPT_PARTS for part in path.parts):
            continue
        items.append((path.relative_to(chkpt).as_posix(), str(path)))
    for path in sorted(chkpt.rglob("*.pth")):
        if any(part in SKIP_CKPT_PARTS for part in path.parts):
            continue
        rel = path.relative_to(chkpt).as_posix()
        if all(rel != label for label, _ in items):
            items.append((rel, str(path)))
    return items


def latest_model_value(items: list[tuple[str, str]]) -> str | None:
    if not items:
        return None
    preferred = [
        value
        for label, value in items
        if not any(tag in label.lower() for tag in ("old", "segment", "trash"))
    ]
    pool = preferred or [value for _, value in items]
    return pool[-1]


def scan_repo_speakers(repo: str | Path) -> list[tuple[str, str]]:
    singer = Path(repo) / "data_svc" / "singer"
    if not singer.is_dir():
        return []
    return [(speaker_label(path), str(path)) for path in sorted(singer.glob("*.npy"))]


def scan_repo_configs(repo: str | Path) -> list[tuple[str, str]]:
    cfg = Path(repo) / "configs"
    items: list[tuple[str, str]] = []
    if cfg.is_dir():
        for path in sorted(cfg.glob("*.yaml")):
            items.append((path.relative_to(Path(repo)).as_posix(), str(path)))
    preferred = next((value for label, value in items if label == "configs/base.yaml"), None)
    if preferred:
        items = [item for item in items if item[1] != preferred]
        items.insert(0, ("configs/base.yaml", preferred))
    return items


def default_config_value(items: list[tuple[str, str]]) -> str | None:
    if not items:
        return None
    for label, value in items:
        if label == "configs/base.yaml":
            return value
    return items[0][1]


def scan_work_csvs(work_dir) -> list[tuple[str, str]]:
    text = as_path(work_dir)
    if not text:
        return []
    folder = Path(text)
    if not folder.is_dir():
        return []
    items: list[tuple[str, str]] = []
    for path in sorted(folder.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True):
        mark = "" if prd_beside(path).is_file() else "  [prdなし]"
        items.append((path.name + mark, str(path)))
    return items


def selected_csv_prd_status(csv_selected, work_dir=None) -> str:
    raw = as_path(csv_selected)
    if not raw:
        return ""
    path = Path(raw)
    folder = as_path(work_dir)
    if folder and (not path.is_file() or is_temp_path(path)):
        local = Path(folder) / path.name
        if local.is_file():
            path = local
    if not path.is_file():
        return ""
    prd = prd_beside(path)
    if prd.is_file():
        return f"prd あり  —  {prd.name}"
    return f"prd なし  —  {prd.name} が見つかりません。Stage 1 で作り直してください"


def dropdown_update(items: list[tuple[str, str]], value: str | None = None):
    if not items:
        return gr.update(choices=[], value=None)
    values = [item[1] for item in items]
    if value and value in values:
        chosen = value
    else:
        chosen = value if value else items[0][1]
        if chosen not in values:
            chosen = items[0][1]
    return gr.update(choices=items, value=chosen)


def empty_mix_rows(speakers: list[tuple[str, str]]) -> list[list]:
    rows = []
    for i, (label, _) in enumerate(speakers):
        rows.append([label, "1.0" if i == 0 else "0"])
    if not rows:
        rows = [["", "0"]]
    return rows


def missing_weight_messages(method: str) -> list[str]:
    lines: list[str] = []
    if method == "crepe":
        path = ROOT / "crepe" / "assets" / "full.pth"
        if not path.is_file() or path.stat().st_size < 1_000_000:
            lines.append(
                f"CREPE の重みがありません: {path}\n"
                "README の Setup にある crepe full.pth（約 85MB）を crepe/assets/ に置いてください。"
            )
    if method == "rmvpe":
        path = ROOT / "pretrain" / "model.pt"
        if not path.is_file() or path.stat().st_size < 1_000_000:
            lines.append(
                f"RMVPE の重みがありません: {path}\n"
                "rmvpe.pt を pretrain/model.pt として置いてください。"
            )
    return lines


# ---------------------------------------------------------------------------
# subprocess / CLI
# ---------------------------------------------------------------------------

def run_python(script_rel: str, args: list[str]) -> str:
    import subprocess

    script = ROOT / script_rel
    cmd = [sys.executable, str(script), *args]
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    blob = "\n".join(part for part in (out, err) if part)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{script_rel} が失敗しました (exit {proc.returncode})\n"
            f"cmd: {format_cli(cmd)}\n\n{blob}"
        )
    return blob


def format_cli(cmd: list[str]) -> str:
    parts: list[str] = []
    for item in cmd:
        if " " in item or any(ch in item for ch in "()&"):
            parts.append(f'"{item}"')
        else:
            parts.append(item)
    if len(parts) <= 3:
        return " ".join(parts)
    head, *rest = parts
    lines = [head + " " + rest[0] + " `"]
    i = 1
    while i < len(rest):
        token = rest[i]
        nxt = rest[i + 1] if i + 1 < len(rest) else None
        if token.startswith("-") and nxt is not None and not nxt.startswith("-"):
            lines.append(f"  {token} {nxt} `")
            i += 2
        else:
            lines.append(f"  {token} `")
            i += 1
    if lines[-1].endswith(" `"):
        lines[-1] = lines[-1][:-2]
    return "\n".join(lines)


_PITCH_FLAGS: set[str] | None = None


def pitch_supported_flags() -> set[str]:
    """このフォルダの pitch/inference.py が受け取れるオプション。fork ごとに違う。"""
    global _PITCH_FLAGS
    if _PITCH_FLAGS is not None:
        return _PITCH_FLAGS
    import subprocess

    proc = subprocess.run(
        [sys.executable, str(ROOT / "pitch" / "inference.py"), "-h"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(ROOT),
    )
    blob = (proc.stdout or "") + (proc.stderr or "")
    known = (
        "--fill-gaps",
        "--no-gap-fill",
        "--post",
        "--rmvpe-harmonic",
        "--rmvpe-bandpass",
        "--max-gap-ms",
        "--uv-th",
        "--smooth",
        "--device",
        "--gate-db",
        "--fmin",
        "--fmax",
        "--softness",
        "--method",
    )
    _PITCH_FLAGS = {flag for flag in known if flag in blob}
    return _PITCH_FLAGS


def apply_gap_fill_to_csv(csv_path: str, max_gap_ms: float) -> None:
    from pitch import load_csv_pitch
    from pitch.inference import _fill_short_gaps, _write_csv_1col

    f0 = np.asarray(load_csv_pitch(csv_path), dtype=np.float32)
    f0 = _fill_short_gaps(f0, sr=16000, hop=160, max_gap_ms=max_gap_ms)
    _write_csv_1col(csv_path, f0)


def pitch_cli_args(
    wav: str,
    pit: str,
    method: str,
    uv_th: float,
    smooth: int,
    device: str,
    post: bool,
    gate_db: float,
    fmin: float,
    fmax: float,
    softness: float,
    max_gap_ms: float,
    fill_gaps: bool,
    rmvpe_harmonic: bool,
    rmvpe_bandpass: bool,
) -> list[str]:
    flags = pitch_supported_flags()
    args = ["-w", wav, "-p", pit]
    if "--method" in flags:
        args += ["--method", method]
    if "--uv-th" in flags:
        args += ["--uv-th", repr(uv_th) if uv_th != 0 else "0"]
    if "--smooth" in flags:
        args += ["--smooth", str(smooth)]
    if "--device" in flags:
        args += ["--device", device]
    if "--gate-db" in flags:
        args += ["--gate-db", str(gate_db)]
    if "--fmin" in flags:
        args += ["--fmin", str(fmin)]
    if "--fmax" in flags:
        args += ["--fmax", str(fmax)]
    if "--softness" in flags:
        args += ["--softness", str(softness)]
    if "--max-gap-ms" in flags:
        args += ["--max-gap-ms", str(max_gap_ms)]
    if post and "--post" in flags:
        args.append("--post")
    if "--fill-gaps" in flags:
        if fill_gaps:
            args.append("--fill-gaps")
        elif "--no-gap-fill" in flags:
            args.append("--no-gap-fill")
    if rmvpe_harmonic and "--rmvpe-harmonic" in flags:
        args.append("--rmvpe-harmonic")
    if rmvpe_bandpass and "--rmvpe-bandpass" in flags:
        args.append("--rmvpe-bandpass")
    return args


# ---------------------------------------------------------------------------
# plots / model
# ---------------------------------------------------------------------------

def plot_pitch(csv_path: str, prd_file: Path | None = None):
    from pitch import load_csv_pitch

    f0 = np.asarray(load_csv_pitch(csv_path), dtype=np.float32)
    if f0.size == 0:
        return None

    t = np.arange(len(f0)) * 0.01
    voiced = f0.copy()
    voiced[voiced <= 0] = np.nan

    has_prd = prd_file is not None and prd_file.is_file()
    fig, axes = plt.subplots(
        2 if has_prd else 1,
        1,
        figsize=(11, 5.2 if has_prd else 3.2),
        sharex=True,
        constrained_layout=True,
    )
    if not has_prd:
        axes = [axes]

    axes[0].plot(t, voiced, color="#0f766e", linewidth=1.1)
    axes[0].set_ylabel("F0 (Hz)")
    axes[0].set_title(Path(csv_path).name)
    axes[0].grid(True, alpha=0.25)
    finite = voiced[np.isfinite(voiced)]
    if finite.size:
        axes[0].set_ylim(max(0.0, float(np.min(finite)) - 20), float(np.max(finite)) + 20)

    if has_prd:
        prd = np.load(prd_file).astype(np.float32).reshape(-1)
        n = min(len(prd), len(t))
        axes[1].plot(t[:n], prd[:n], color="#7c3aed", linewidth=1.0)
        axes[1].set_ylabel("periodicity")
        axes[1].set_ylim(-0.02, 1.05)
        axes[1].grid(True, alpha=0.25)

    axes[-1].set_xlabel("time (s)")
    fig.patch.set_facecolor("#f8fafc")
    return fig


def unload_model():
    import torch

    _model_cache["model"] = None
    _model_cache["path"] = None
    _model_cache["config"] = None
    _model_cache["hp"] = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_model(model_path: str, config_path: str, device_name: str):
    import torch
    from omegaconf import OmegaConf
    from vits.models import SynthesizerInfer
    from svc_inference import load_svc_model

    model_abs = str(Path(model_path).resolve())
    config_abs = str(Path(config_path).resolve())
    if (
        _model_cache["model"] is not None
        and _model_cache["path"] == model_abs
        and _model_cache["config"] == config_abs
        and _model_cache["device"] == device_name
    ):
        return _model_cache["model"], _model_cache["hp"], torch.device(device_name)

    unload_model()
    hp = OmegaConf.load(config_abs)
    model = SynthesizerInfer(
        hp.data.filter_length // 2 + 1,
        hp.data.segment_size // hp.data.hop_length,
        hp,
    )
    load_svc_model(model_abs, model)
    device = torch.device(
        device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu"
    )
    model.eval().to(device)
    _model_cache.update(
        model=model,
        path=model_abs,
        config=config_abs,
        hp=hp,
        device=str(device),
    )
    return model, hp, device


def extract_ppg_vec(wav: str, work_dir: Path, reuse: bool, force: bool, log: list[str], progress, start: float):
    ppg_path = cache_feature_path(work_dir, wav, ".ppg.npy")
    vec_path = cache_feature_path(work_dir, wav, ".vec.npy")

    if force or not (reuse and ppg_path.is_file()):
        log.append("PPG (whisper) を抽出しています…")
        progress(start, desc="PPG (whisper)")
        run_python("whisper/inference.py", ["-w", wav, "-p", str(ppg_path)])
    else:
        log.append(f"PPG を再利用: {ppg_path}")

    if force or not (reuse and vec_path.is_file()):
        log.append("HuBERT を抽出しています…")
        progress(start + 0.15, desc="HuBERT")
        run_python("hubert/inference.py", ["-w", wav, "-v", str(vec_path)])
    else:
        log.append(f"HuBERT を再利用: {vec_path}")

    return ppg_path, vec_path


def make_retrieval(enable, repo: Path, spk_path: str, ratio, n_vectors, prefix):
    from feature_retrieval import DummyRetrieval, FaissIndexRetrieval, load_retrieve_index
    from svc_inference import get_speaker_name_from_path

    if not enable:
        return DummyRetrieval()
    speaker_name = get_speaker_name_from_path(Path(spk_path))
    base = repo / "data_svc" / "indexes" / speaker_name
    hubert = base / f"{prefix or ''}hubert.index"
    whisper = base / f"{prefix or ''}whisper.index"
    return FaissIndexRetrieval(
        hubert_index=load_retrieve_index(filepath=hubert, ratio=float(ratio), n_nearest_vectors=int(n_vectors)),
        whisper_index=load_retrieve_index(filepath=whisper, ratio=float(ratio), n_nearest_vectors=int(n_vectors)),
    )


# ---------------------------------------------------------------------------
# speaker mix
# ---------------------------------------------------------------------------

def iter_mix_rows(df):
    if df is None:
        return
    if hasattr(df, "values"):
        rows = df.values.tolist()
    else:
        rows = list(df)
    for row in rows:
        if not row:
            continue
        name = str(row[0]).strip() if row[0] is not None else ""
        if not name:
            continue
        try:
            weight = parse_weight(row[1])
        except Exception:
            weight = 0.0
        yield name, weight


def mix_sum_status(df):
    total = sum(weight for _, weight in iter_mix_rows(df) if weight > 0)
    if abs(total - 1.0) <= 0.001:
        return f"合計 {total:.3f}  ✓"
    return f"合計 {total:.3f}  ← 1.0 にしてください"


def normalize_mix_df(df):
    rows = [[name, weight] for name, weight in iter_mix_rows(df)]
    total = sum(weight for _, weight in rows if weight > 0)
    if total <= 0:
        return rows, "重みが全部 0 です"
    out = []
    for name, weight in rows:
        out.append([name, format_weight((weight / total) if weight > 0 else 0.0)])
    return out, mix_sum_status(out)


def resolve_speaker_path(repo: Path, name_or_path: str) -> Path:
    raw = as_path(name_or_path)
    if not raw:
        raise FileNotFoundError("話者が空です。")
    path = Path(raw)
    if path.is_file():
        return path
    singer = repo / "data_svc" / "singer"
    for candidate in (
        singer / raw,
        singer / f"{raw}.npy",
        singer / f"{raw}.spk.npy",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"話者ファイルが見つかりません: {raw}（{singer}）")


def build_speaker(repo: Path, work_dir: Path, use_mix: bool, spk_drop, mix_df) -> tuple[str, str, np.ndarray]:
    """
    Returns (spk_path_for_log, speaker_tag_for_filename, embedding)
    """
    if use_mix:
        pairs = []
        for name, weight in iter_mix_rows(mix_df):
            if weight > 0:
                pairs.append((resolve_speaker_path(repo, name), weight))
        if not pairs:
            raise ValueError("ミックスする話者と、0より大きい重みを入れてください。")
        total = sum(weight for _, weight in pairs)
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"話者ミックスの合計が {total:.3f} です。1.0 にしてください（「1.0に正規化」が使えます）。")
        vec = None
        tags = []
        for path, weight in pairs:
            arr = np.load(path).astype(np.float32)
            vec = arr * float(weight) if vec is None else vec + arr * float(weight)
            tags.append(f"{speaker_label(path)}{format_weight(weight)}")
        tag = "+".join(tags) if len(tags) > 1 else tags[0]
        mix_path = work_dir / f"_mix_{sanitize_filename(tag)}.spk.npy"
        np.save(mix_path, vec.astype(np.float32), allow_pickle=False)
        return str(mix_path), tag, vec.astype(np.float32)

    path = resolve_speaker_path(repo, as_path(spk_drop) or "")
    return str(path), speaker_label(path), np.load(path).astype(np.float32)


# ---------------------------------------------------------------------------
# UI callbacks
# ---------------------------------------------------------------------------

def apply_work_dir(work_dir, selected_csv=None):
    try:
        folder = require_work_dir(work_dir)
        save_settings(work_dir=str(folder))
        items = scan_work_csvs(folder)
        status = f"{folder}　CSV {len(items)} 個"
        drop = dropdown_update(items, selected_csv)
        chosen = drop.get("value") if isinstance(drop, dict) else selected_csv
        return str(folder), status, drop, selected_csv_prd_status(chosen, folder)
    except Exception as exc:
        return work_dir or "", f"作業フォルダを開けません: {exc}", gr.update(), ""


def browse_work_dir(current, selected_csv=None):
    chosen = pick_directory(current)
    if not chosen:
        return current or "", "キャンセルしました", gr.update(), gr.update()
    return apply_work_dir(chosen, selected_csv)


def open_work_dir(work_dir):
    try:
        folder = require_work_dir(work_dir)
        if sys.platform.startswith("win"):
            os.startfile(folder)  # type: ignore[attr-defined]
        else:
            import subprocess

            subprocess.Popen(["xdg-open", str(folder)])
        return f"開きました: {folder}"
    except Exception as exc:
        return f"フォルダを開けません: {exc}"


def add_repo(extra_path, current_repo):
    raw = as_path(extra_path)
    if not raw:
        return gr.update(), "追加するフォルダのパスを入れてください。"
    path = Path(raw)
    if not is_svc_repo(path):
        return gr.update(), f"chkpt が見つかりません: {path}"
    settings = load_settings()
    extras = list(settings.get("extra_repos") or [])
    resolved = str(path.resolve())
    if resolved not in extras:
        extras.append(resolved)
    save_settings(extra_repos=extras, last_repo=resolved)
    repos = scan_repos(extras)
    return dropdown_update(repos, resolved), f"追加しました: {repo_short_name(path)}"


def on_repo_change(repo_value):
    repo = as_path(repo_value) or str(ROOT)
    save_settings(last_repo=repo)
    models = scan_repo_models(repo)
    speakers = scan_repo_speakers(repo)
    configs = scan_repo_configs(repo)
    mix_visible = len(speakers) >= 2
    mix_rows = empty_mix_rows(speakers)
    status = (
        f"{Path(repo).name}　モデル {len(models)} / 話者 {len(speakers)}"
        + ("　→ 複数話者なのでミックスできます" if mix_visible else "")
    )
    return (
        dropdown_update(models, latest_model_value(models)),
        dropdown_update(speakers, speakers[0][1] if speakers else None),
        dropdown_update(configs, default_config_value(configs)),
        gr.update(visible=mix_visible),
        gr.update(value=False),
        mix_rows,
        mix_sum_status(mix_rows),
        status,
    )


def preview_names(
    wav, work_dir, method, uv_th, post, harmonic, fill_gaps, max_gap_ms,
    csv_override, csv_selected, spk_drop, use_mix, mix_df,
):
    try:
        folder = as_path(work_dir) or "(作業フォルダ未設定)"
        wav_path = as_path(wav)
        stem_src = Path(wav_path).name if wav_path else "(入力WAV)"
        try:
            uv = parse_float(uv_th, "uv-th")
        except Exception:
            uv = 0.000015
        try:
            gap = parse_float(max_gap_ms, "max-gap-ms")
        except Exception:
            gap = 3.0
        if as_path(csv_override):
            csv_name = Path(as_path(csv_override)).name
            if not csv_name.lower().endswith(".csv"):
                csv_name += ".csv"
        elif wav_path:
            csv_name = default_csv_name(
                wav_path, method or "crepe", uv_th, uv, bool(post), bool(harmonic),
                bool(fill_gaps), gap,
            )
        elif as_path(csv_selected):
            csv_name = Path(as_path(csv_selected)).name
        else:
            csv_name = "(CSV)"
        if use_mix:
            names = [
                f"{name}{format_weight(weight)}"
                for name, weight in iter_mix_rows(mix_df)
                if weight > 0
            ]
            speaker = "+".join(names) if names else "mix"
        else:
            speaker = speaker_label(as_path(spk_drop) or "speaker")
        wav_name = default_wav_name(csv_name, speaker)
        return f"CSV → `{csv_name}`\nWAV → `{wav_name}`\n保存先 → `{folder}`\n入力 → `{stem_src}`"
    except Exception as exc:
        return f"プレビューを作れません: {exc}"


def collect_stage1_params(
    method, uv_th, max_gap_ms, fill_gaps, post, device,
    smooth, gate_db, fmin, fmax, softness,
    rmvpe_harmonic, rmvpe_bandpass,
):
    return {
        "method": method,
        "uv_th": parse_float(uv_th, "uv-th"),
        "uv_text": str(uv_th).strip(),
        "max_gap_ms": parse_float(max_gap_ms, "max-gap-ms"),
        "fill_gaps": bool(fill_gaps),
        "post": bool(post),
        "device": device if device in ("cuda", "cpu") else DEFAULT_DEVICE,
        "smooth": parse_int(smooth, "smooth"),
        "gate_db": parse_float(gate_db, "gate-db"),
        "fmin": parse_float(fmin, "fmin"),
        "fmax": parse_float(fmax, "fmax"),
        "softness": parse_float(softness, "softness"),
        "rmvpe_harmonic": bool(rmvpe_harmonic),
        "rmvpe_bandpass": bool(rmvpe_bandpass),
    }


def run_stage1(
    wav_uploaded,
    work_dir,
    csv_override,
    method,
    uv_th,
    max_gap_ms,
    fill_gaps,
    post,
    device,
    smooth,
    gate_db,
    fmin,
    fmax,
    softness,
    rmvpe_harmonic,
    rmvpe_bandpass,
    overwrite,
    progress=gr.Progress(),
):
    empty_plot = None
    try:
        folder = require_work_dir(work_dir)
        save_settings(work_dir=str(folder))
        wav = materialize_wav(wav_uploaded, folder)
        p = collect_stage1_params(
            method, uv_th, max_gap_ms, fill_gaps, post, device,
            smooth, gate_db, fmin, fmax, softness,
            rmvpe_harmonic, rmvpe_bandpass,
        )
        override = as_path(csv_override)
        if override:
            name = Path(override).name
            if not name.lower().endswith(".csv"):
                name += ".csv"
            pit_path = folder / sanitize_filename(name)
        else:
            pit_path = folder / default_csv_name(
                wav, p["method"], p["uv_text"], p["uv_th"], p["post"], p["rmvpe_harmonic"],
                p["fill_gaps"], p["max_gap_ms"],
            )
        pit_path = unique_path(pit_path, overwrite)
        missing = missing_weight_messages(p["method"])
        if missing:
            raise FileNotFoundError("\n".join(missing))

        args = pitch_cli_args(
            wav, str(pit_path), p["method"], p["uv_th"], p["smooth"], p["device"],
            p["post"], p["gate_db"], p["fmin"], p["fmax"], p["softness"],
            p["max_gap_ms"], p["fill_gaps"], p["rmvpe_harmonic"], p["rmvpe_bandpass"],
        )
        cli = format_cli([sys.executable, str(ROOT / "pitch" / "inference.py"), *args])

        progress(0.15, desc="ピッチ抽出中（CREPE/RMVPE）")
        blob = run_python("pitch/inference.py", args)
        extra_note = ""
        flags = pitch_supported_flags()
        if p["fill_gaps"] and "--fill-gaps" not in flags and not p["post"]:
            apply_gap_fill_to_csv(str(pit_path), p["max_gap_ms"])
            extra_note = (
                f"このforkの pitch/inference.py に --fill-gaps が無いので、"
                f"UI側で max-gap {p['max_gap_ms']:g}ms を適用しました。"
            )
        progress(0.9, desc="プロット作成")

        prd = prd_beside(pit_path)
        if not prd.is_file():
            raise FileNotFoundError(f"periodicity が保存されていません: {prd}")

        fig = plot_pitch(str(pit_path), prd)
        log = [
            "Stage 1 完了",
            f"wav : {wav}",
            f"pit : {pit_path}",
            f"prd : {prd}",
        ]
        if extra_note:
            log.extend(["", extra_note])
        log.extend(["", blob])
        items = scan_work_csvs(folder)
        return (
            "\n".join(log),
            cli,
            dropdown_update(items, str(pit_path)),
            f"prd OK  ({prd.name})",
            fig,
            f"作業フォルダ: {folder}　CSV {len(items)} 個",
        )
    except Exception as exc:
        return (
            f"Stage 1 失敗\n{exc}\n\n{traceback.format_exc()}",
            "",
            gr.update(),
            "prd なし",
            empty_plot,
            gr.update(),
        )


def inspect_pitch(csv_selected, work_dir):
    try:
        csv_path, prd = resolve_csv_and_prd(csv_selected, work_dir)
        fig = plot_pitch(str(csv_path), prd)
        from pitch import load_csv_pitch

        f0 = np.asarray(load_csv_pitch(str(csv_path)), dtype=np.float32)
        voiced = f0[f0 > 0]
        summary = (
            f"{csv_path}\nprd: {prd}\n"
            f"frames={len(f0)}  voiced={len(voiced)}"
            + (f"  F0={voiced.min():.1f}–{voiced.max():.1f} Hz" if voiced.size else "")
        )
        return fig, summary, f"prd OK  ({prd.name})"
    except Exception as exc:
        return None, f"読み込み失敗: {exc}", "prd なし"


def run_stage2(
    wav_uploaded,
    work_dir,
    csv_selected,
    repo_value,
    model_drop,
    config_drop,
    spk_drop,
    use_mix,
    mix_df,
    shift,
    out_override,
    write_out_pit,
    reuse_features,
    force_features,
    enable_retrieval,
    retrieval_ratio,
    n_retrieval_vectors,
    retrieval_prefix,
    overwrite,
    progress=gr.Progress(),
):
    try:
        import torch
        from scipy.io.wavfile import write as wav_write
        from pitch import load_csv_pitch
        from svc_inference import maybe_prepare_extra_features, svc_infer, ensure_parent_dir

        folder = require_work_dir(work_dir)
        save_settings(work_dir=str(folder))
        wav = materialize_wav(wav_uploaded, folder)
        csv_path, prd = resolve_csv_and_prd(csv_selected, folder)

        repo = Path(as_path(repo_value) or ROOT)
        if not repo.is_dir():
            raise FileNotFoundError(f"学習フォルダが見つかりません: {repo}")

        model_path = as_path(model_drop)
        config_path = as_path(config_drop)
        if not model_path or not Path(model_path).is_file():
            raise FileNotFoundError("モデル (.pt) を選んでください。")
        if not config_path or not Path(config_path).is_file():
            raise FileNotFoundError("config を選んでください。")

        spk_path, speaker_tag, spk_np = build_speaker(repo, folder, bool(use_mix), spk_drop, mix_df)
        shift_i = parse_int(shift, "shift")

        override = as_path(out_override)
        if override:
            name = Path(override).name
            if not name.lower().endswith(".wav"):
                name += ".wav"
            out_path = folder / sanitize_filename(name)
        else:
            out_path = folder / default_wav_name(csv_path, speaker_tag)
        out_path = unique_path(out_path, overwrite)
        out_pit_path = out_path.with_name(out_path.stem + "_pit.wav")

        log = [
            f"wav    : {wav}",
            f"pit    : {csv_path}",
            f"prd    : {prd}",
            f"repo   : {repo}",
            f"model  : {model_path}",
            f"config : {config_path}",
            f"spk    : {spk_path}",
            f"speaker: {speaker_tag}",
            f"shift  : {shift_i}",
        ]

        t0 = time.perf_counter()
        progress(0.05, desc="モデル読み込み")
        model, hp, device = get_model(model_path, config_path, DEFAULT_DEVICE)
        log.append(f"device : {device}")
        log.append(f"model load: {time.perf_counter() - t0:.1f}s")

        t1 = time.perf_counter()
        ppg_path, vec_path = extract_ppg_vec(
            wav, folder, bool(reuse_features), bool(force_features), log, progress, 0.2
        )
        log.append(f"PPG/HuBERT: {time.perf_counter() - t1:.1f}s")

        cli_args = [
            sys.executable, str(ROOT / "svc_inference.py"),
            "--config", config_path,
            "--model", model_path,
            "--wave", wav,
            "--spk", spk_path,
            "--pit", str(csv_path),
            "--ppg", str(ppg_path),
            "--vec", str(vec_path),
            "--shift", str(shift_i),
            "--out", str(out_path),
            "--out-pit", str(out_pit_path),
        ]
        cli = format_cli(cli_args)

        progress(0.55, desc="Acoustic Dynamics 特徴量")
        ns = SimpleNamespace(pit=str(csv_path), wave=wav, eng=None, deng=None, flat=None)
        eng_np, deng_np, prd_np, flat_np = maybe_prepare_extra_features(ns)
        retrieval = make_retrieval(
            enable_retrieval, repo, spk_path, retrieval_ratio, n_retrieval_vectors, retrieval_prefix
        )

        ppg = torch.FloatTensor(np.repeat(np.load(ppg_path), 2, 0))
        vec = torch.FloatTensor(np.repeat(np.load(vec_path), 2, 0))
        pit_t = torch.FloatTensor(np.asarray(load_csv_pitch(str(csv_path)), dtype=np.float32))
        if shift_i != 0:
            pit_t = pit_t * (2 ** (shift_i / 12.0))
        spk_t = torch.FloatTensor(spk_np)
        eng = torch.FloatTensor(eng_np)
        deng = torch.FloatTensor(deng_np)
        prd_t = torch.FloatTensor(prd_np)
        flat = torch.FloatTensor(flat_np)

        progress(0.7, desc="変換")
        t2 = time.perf_counter()
        out_audio = svc_infer(
            model, retrieval, spk_t, pit_t, ppg, vec, eng, deng, prd_t, flat,
            hp, device, str(out_pit_path),
        )
        ensure_parent_dir(str(out_path))
        wav_write(str(out_path), hp.data.sampling_rate, out_audio)
        log.append(f"convert : {time.perf_counter() - t2:.1f}s")
        log.append(f"total   : {time.perf_counter() - t0:.1f}s")

        log.extend(["", f"出力 : {out_path}", f"out-pit : {out_pit_path}", "Stage 2 完了"])
        progress(1.0, desc="完了")
        pit_audio = str(out_pit_path) if write_out_pit and out_pit_path.is_file() else None
        return (
            "\n".join(log),
            cli,
            serve_for_gradio(out_path),
            serve_for_gradio(pit_audio),
            str(out_path),
        )
    except Exception as exc:
        return (
            f"Stage 2 失敗\n{exc}\n\n{traceback.format_exc()}",
            "",
            None,
            None,
            as_path(out_override) or "",
        )


def run_pipeline(
    wav_uploaded, work_dir, csv_override,
    method, uv_th, max_gap_ms, fill_gaps, post, device,
    smooth, gate_db, fmin, fmax, softness,
    rmvpe_harmonic, rmvpe_bandpass, overwrite,
    csv_selected, repo_value, model_drop, config_drop, spk_drop, use_mix, mix_df,
    shift, out_override, write_out_pit,
    reuse_features, force_features,
    enable_retrieval, retrieval_ratio, n_retrieval_vectors, retrieval_prefix,
    progress=gr.Progress(),
):
    s1 = run_stage1(
        wav_uploaded, work_dir, csv_override,
        method, uv_th, max_gap_ms, fill_gaps, post, device,
        smooth, gate_db, fmin, fmax, softness,
        rmvpe_harmonic, rmvpe_bandpass, overwrite,
        progress=progress,
    )
    s1_log, s1_cli, csv_upd, prd_status, fig, work_status = s1
    pit_used = None
    for line in str(s1_log).splitlines():
        if line.startswith("pit : "):
            pit_used = line[6:].strip()
            break
    if pit_used and Path(str(pit_used)).is_file() and "Stage 1 完了" in s1_log:
        s2 = run_stage2(
            wav_uploaded, work_dir, pit_used,
            repo_value, model_drop, config_drop, spk_drop, use_mix, mix_df,
            shift, out_override, write_out_pit,
            reuse_features, force_features,
            enable_retrieval, retrieval_ratio, n_retrieval_vectors, retrieval_prefix,
            overwrite, progress=progress,
        )
        s2_log, s2_cli, out_wav, out_pit, out_box = s2
        return (
            s1_log + "\n\n====\n\n" + s2_log,
            s1_cli + "\n\n" + s2_cli,
            csv_upd, prd_status, fig, work_status,
            s2_log, s2_cli, out_wav, out_pit, out_box,
        )
    return (
        s1_log, s1_cli, csv_upd, prd_status, fig, work_status,
        "Stage 1 が失敗したため変換していません。",
        "", None, None, "",
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

CSS = """
.gradio-container { max-width: 980px !important; }
#go-1, #go-2 { font-weight: 650; }
footer { display: none !important; }
"""

HELP_MD = """
作業フォルダに CSV / prd / WAV が全部出ます。

- CSV: `曲名_crepe_uv0.000015.csv`
- WAV: `そのCSV名_話者名.wav`
- ミックス: `そのCSV名_Rei_Adachi0.25+MUGICA0.75.wav`

ミックスの重みは小数（`0.25`）で、合計は 1.0 です。
"""


def build_ui():
    settings = load_settings()
    repos = scan_repos(settings.get("extra_repos"))
    repo_values = [value for _, value in repos]
    last_repo = settings.get("last_repo") if settings.get("last_repo") in repo_values else (
        repos[0][1] if repos else str(ROOT)
    )
    models = scan_repo_models(last_repo)
    speakers = scan_repo_speakers(last_repo)
    configs = scan_repo_configs(last_repo)
    work_dir_init = settings.get("work_dir") or ""
    work_csvs = scan_work_csvs(work_dir_init)
    mix_visible = len(speakers) >= 2
    mix_rows = empty_mix_rows(speakers)

    theme = gr.themes.Soft(primary_hue="teal", secondary_hue="slate", neutral_hue="slate")

    with gr.Blocks(title="Acoustic Dynamics", theme=theme, css=CSS) as demo:
        gr.Markdown(f"# Acoustic Dynamics　`{DEFAULT_DEVICE}`")

        with gr.Row():
            btn_browse_work = gr.Button("作業フォルダ…", scale=1)
            work_dir = gr.Textbox(
                value=work_dir_init,
                show_label=False,
                placeholder="作業フォルダ",
                scale=4,
                interactive=False,
            )
            btn_open_work = gr.Button("開く", scale=1)
        work_status = gr.Markdown(
            f"{work_dir_init}" if work_dir_init else "作業フォルダを選んでください"
        )
        wav_in = gr.Audio(label="入力WAV", sources=["upload"], type="filepath")
        name_preview = gr.Markdown("")

        with gr.Tabs():
            with gr.Tab("ピッチ"):
                with gr.Row():
                    method = gr.Radio(choices=["crepe", "rmvpe"], value="crepe", label="method")
                    device = gr.Radio(choices=["cuda", "cpu"], value=DEFAULT_DEVICE, label="device")
                with gr.Row():
                    uv_th = gr.Textbox(value="0.000015", label="uv-th", scale=2)
                    post = gr.Checkbox(value=False, label="post")
                    fill_gaps = gr.Checkbox(value=False, label="max-gap")
                    max_gap_ms = gr.Number(value=3, label="ms", precision=1, scale=1)
                csv_drop = gr.Dropdown(
                    label="作業フォルダのCSV",
                    choices=work_csvs,
                    value=work_csvs[0][1] if work_csvs else None,
                )
                with gr.Row():
                    btn_stage1 = gr.Button("抽出", variant="primary", elem_id="go-1")
                    btn_inspect = gr.Button("プロット")
                f0_plot = gr.Plot(label="F0")
                prd_status = gr.Textbox(
                    label="prd",
                    value=selected_csv_prd_status(work_csvs[0][1] if work_csvs else None, work_dir_init),
                    interactive=False,
                    max_lines=1,
                )
                with gr.Accordion("詳細 / ログ", open=False):
                    csv_override = gr.Textbox(label="CSV名（空なら自動）")
                    with gr.Row():
                        smooth = gr.Number(value=1, label="smooth", precision=0)
                        gate_db = gr.Number(value=-42.0, label="gate-db")
                        softness = gr.Number(value=25.0, label="softness")
                        fmin = gr.Number(value=40.0, label="fmin")
                        fmax = gr.Number(value=1200.0, label="fmax")
                    with gr.Row():
                        rmvpe_harmonic = gr.Checkbox(value=False, label="rmvpe-harmonic")
                        rmvpe_bandpass = gr.Checkbox(value=False, label="rmvpe-bandpass")
                    s1_log = gr.Textbox(label="ログ", lines=6, interactive=False)
                    s1_cli = gr.Textbox(label="コマンド", lines=4, interactive=False)

            with gr.Tab("変換"):
                with gr.Row():
                    repo_drop = gr.Dropdown(
                        label="学習フォルダ",
                        choices=repos,
                        value=last_repo,
                        allow_custom_value=True,
                        scale=2,
                    )
                    model_drop = gr.Dropdown(
                        label="モデル",
                        choices=models,
                        value=latest_model_value(models),
                        allow_custom_value=True,
                        scale=2,
                    )
                with gr.Row():
                    spk_drop = gr.Dropdown(
                        label="話者",
                        choices=speakers,
                        value=speakers[0][1] if speakers else None,
                        allow_custom_value=True,
                    )
                    shift = gr.Slider(-24, 24, value=0, step=1, label="shift")
                repo_status = gr.Markdown(
                    f"{Path(last_repo).name}　モデル {len(models)} / 話者 {len(speakers)}"
                )
                with gr.Group(visible=mix_visible) as mix_group:
                    use_mix = gr.Checkbox(value=False, label="話者ミックス（合計 1.0）")
                    mix_df = gr.Dataframe(
                        headers=["話者", "重み"],
                        datatype=["str", "str"],
                        value=mix_rows,
                        interactive=True,
                    )
                    with gr.Row():
                        mix_status = gr.Textbox(value=mix_sum_status(mix_rows), label="合計", interactive=False)
                        btn_norm_mix = gr.Button("1.0に正規化")
                with gr.Row():
                    btn_stage2 = gr.Button("変換", variant="primary", elem_id="go-2")
                    btn_both = gr.Button("抽出して変換")
                out_audio = gr.Audio(label="結果", type="filepath")
                out_path_box = gr.Textbox(label="保存先", interactive=False)
                with gr.Accordion("詳細 / ログ", open=False):
                    config_drop = gr.Dropdown(
                        label="config",
                        choices=configs,
                        value=default_config_value(configs),
                        allow_custom_value=True,
                    )
                    out_override = gr.Textbox(label="出力WAV名（空なら自動）")
                    write_out_pit = gr.Checkbox(value=False, label="デバッグ用ピッチ波形 (out-pit) を出す")
                    pit_audio = gr.Audio(label="out-pit", type="filepath")
                    overwrite = gr.Checkbox(value=False, label="同名を上書き")
                    reuse_features = gr.Checkbox(value=True, label="PPG / HuBERT を再利用")
                    force_features = gr.Checkbox(value=False, label="PPG / HuBERT を再抽出")
                    enable_retrieval = gr.Checkbox(value=False, label="Feature Retrieval")
                    with gr.Row():
                        retrieval_ratio = gr.Slider(0, 1, value=0.5, step=0.05, label="ratio")
                        n_retrieval_vectors = gr.Number(value=3, precision=0, label="n")
                    retrieval_prefix = gr.Textbox(label="index prefix", value="")
                    extra_repo = gr.Textbox(label="学習フォルダを追加（パス）")
                    btn_add_repo = gr.Button("追加")
                    s2_log = gr.Textbox(label="ログ", lines=8, interactive=False)
                    s2_cli = gr.Textbox(label="コマンド", lines=5, interactive=False)

        with gr.Accordion("使い方", open=False):
            gr.Markdown(HELP_MD)

        preview_inputs = [
            wav_in, work_dir, method, uv_th, post, rmvpe_harmonic, fill_gaps, max_gap_ms,
            csv_override, csv_drop, spk_drop, use_mix, mix_df,
        ]
        for comp in preview_inputs:
            comp.change(preview_names, inputs=preview_inputs, outputs=[name_preview])

        btn_browse_work.click(
            browse_work_dir,
            inputs=[work_dir, csv_drop],
            outputs=[work_dir, work_status, csv_drop, prd_status],
        )
        btn_open_work.click(open_work_dir, inputs=[work_dir], outputs=[work_status])
        btn_add_repo.click(add_repo, inputs=[extra_repo, repo_drop], outputs=[repo_drop, repo_status])
        repo_drop.change(
            on_repo_change,
            inputs=[repo_drop],
            outputs=[model_drop, spk_drop, config_drop, mix_group, use_mix, mix_df, mix_status, repo_status],
        )
        mix_df.change(mix_sum_status, inputs=[mix_df], outputs=[mix_status])
        csv_drop.change(selected_csv_prd_status, inputs=[csv_drop, work_dir], outputs=[prd_status])
        btn_norm_mix.click(normalize_mix_df, inputs=[mix_df], outputs=[mix_df, mix_status])

        stage1_inputs = [
            wav_in, work_dir, csv_override, method, uv_th, max_gap_ms, fill_gaps, post, device,
            smooth, gate_db, fmin, fmax, softness, rmvpe_harmonic, rmvpe_bandpass, overwrite,
        ]
        btn_stage1.click(
            run_stage1,
            inputs=stage1_inputs,
            outputs=[s1_log, s1_cli, csv_drop, prd_status, f0_plot, work_status],
        )
        btn_inspect.click(inspect_pitch, inputs=[csv_drop, work_dir], outputs=[f0_plot, s1_log, prd_status])

        stage2_inputs = [
            wav_in, work_dir, csv_drop, repo_drop, model_drop, config_drop,
            spk_drop, use_mix, mix_df, shift, out_override, write_out_pit,
            reuse_features, force_features,
            enable_retrieval, retrieval_ratio, n_retrieval_vectors, retrieval_prefix,
            overwrite,
        ]
        btn_stage2.click(
            run_stage2,
            inputs=stage2_inputs,
            outputs=[s2_log, s2_cli, out_audio, pit_audio, out_path_box],
        )
        btn_both.click(
            run_pipeline,
            inputs=stage1_inputs + [
                csv_drop, repo_drop, model_drop, config_drop, spk_drop, use_mix, mix_df,
                shift, out_override, write_out_pit,
                reuse_features, force_features,
                enable_retrieval, retrieval_ratio, n_retrieval_vectors, retrieval_prefix,
            ],
            outputs=[
                s1_log, s1_cli, csv_drop, prd_status, f0_plot, work_status,
                s2_log, s2_cli, out_audio, pit_audio, out_path_box,
            ],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Acoustic Dynamics 推論 UI")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--listen", action="store_true", help="0.0.0.0 で待受")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    demo = build_ui()
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name="0.0.0.0" if args.listen else "127.0.0.1",
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
        show_error=True,
        allowed_paths=collect_allowed_paths(),
    )


if __name__ == "__main__":
    main()
