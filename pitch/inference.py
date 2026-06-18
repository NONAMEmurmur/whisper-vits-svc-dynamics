import argparse
import os
import sys

import numpy as np
import torch


# repo ルートをパスに追加
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _soft_range_gate(f0, fmin, fmax, softness_hz=25.0):
    """
    範囲外に出たF0をバッサリ0にせず、なだらかに弱めるゲート
    """
    f0 = f0.copy()
    lo = fmin - softness_hz
    hi = fmax + softness_hz

    mask = (f0 > lo) & (f0 < fmin)
    f0[mask] *= (f0[mask] - lo) / (fmin - lo)

    mask = (f0 > fmax) & (f0 < hi)
    f0[mask] *= (hi - f0[mask]) / (hi - fmax)

    f0[(f0 <= lo) | (f0 >= hi)] = 0.0
    return f0


def _fill_short_gaps(f0, sr=16000, hop=160, max_gap_ms=60.0):
    """
    短い無音ギャップだけ線形補間で埋める
    """
    f0 = f0.copy()
    max_gap = int(max_gap_ms / 1000 * sr / hop)

    i = 0
    n = len(f0)
    while i < n:
        if f0[i] != 0:
            i += 1
            continue

        j = i
        while j < n and f0[j] == 0:
            j += 1

        gap = j - i
        if gap <= max_gap and i > 0 and j < n:
            f0[i:j] = np.linspace(f0[i - 1], f0[j], gap + 2)[1:-1]

        i = j
    return f0


def _energy_gate(x: np.ndarray, sr: int = 16000, hop: int = 160, frame: int = 1024, thresh_db: float = -42.0):
    """
    簡易エネルギーゲート（voiced/unvoiced マスクを返す）
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1:
        x = x.reshape(-1)

    pad = frame // 2
    xp = np.pad(x, (pad, pad), mode="reflect")

    n = 1 + (len(xp) - frame) // hop
    rms = np.empty(n, dtype=np.float32)
    for i in range(n):
        seg = xp[i * hop: i * hop + frame]
        rms[i] = np.sqrt(np.mean(seg * seg) + 1e-12)

    db = 20.0 * np.log10(rms + 1e-12)
    voiced = db >= float(thresh_db)
    return voiced


def _smooth_f0(f0: np.ndarray, smooth: int) -> np.ndarray:
    if smooth is None or int(smooth) <= 1:
        return f0.astype(np.float32)
    k = int(smooth)
    kernel = np.ones(k, dtype=np.float32) / k
    pad = k // 2
    f0p = np.pad(f0, (pad, pad), mode="edge")
    return np.convolve(f0p, kernel, mode="valid").astype(np.float32)


def _smooth_scalar_track(x: np.ndarray, smooth: int) -> np.ndarray:
    if smooth is None or int(smooth) <= 1:
        return x.astype(np.float32)
    k = int(smooth)
    kernel = np.ones(k, dtype=np.float32) / k
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, kernel, mode="valid").astype(np.float32)


def _write_csv_1col(out_csv: str, f0: np.ndarray) -> None:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        for v in f0:
            if not np.isfinite(v):
                v = 0.0
            f.write(f"{float(v):.6f}\n")


def _prd_path_from_pitch_csv(out_csv: str) -> str:
    if out_csv.endswith(".csv"):
        return out_csv[:-4] + ".prd.npy"
    return out_csv + ".prd.npy"


def _save_prd_from_pitch_csv_path(out_csv: str, periodicity: np.ndarray) -> str:
    prd_path = _prd_path_from_pitch_csv(out_csv)
    os.makedirs(os.path.dirname(prd_path) or ".", exist_ok=True)
    np.save(prd_path, periodicity.astype(np.float32), allow_pickle=False)
    return prd_path


def load_csv_pitch(path: str):
    """
    private_ui.py が `from pitch import load_csv_pitch` で拾う想定の関数。
    1列CSV（数値のみ）を読んで float の list を返す。
    """
    vals = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            s = line.strip()
            if not s:
                continue
            try:
                vals.append(float(s))
            except ValueError:
                if "," in s:
                    try:
                        vals.append(float(s.split(",")[-1]))
                    except ValueError:
                        vals.append(0.0)
                else:
                    vals.append(0.0)
    return vals


def extract_f0_rmvpe(
    wav_path,
    uv_th=0.000015,
    smooth=1,
    device="cpu",
    post=False,
    gate_db=-42.0,
    fmin=40.0,
    fmax=1200.0,
    softness=25.0,
    max_gap_ms=60.0,
    rmvpe_harmonic=False,
    rmvpe_bandpass=False,
):
    import os
    import sys
    import numpy as np
    import librosa
    import soundfile as sf

    base = ROOT
    sovits_rmvpe_dir = os.path.join(base, "sovits", "rmvpe")
    weight_path = os.path.join(base, "pretrain", "model.pt")

    if sovits_rmvpe_dir not in sys.path:
        sys.path.insert(0, sovits_rmvpe_dir)

    x, sr = sf.read(wav_path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float32, copy=False)
    if sr != 16000:
        x = librosa.resample(x, orig_sr=sr, target_sr=16000).astype(np.float32)
        sr = 16000

    if rmvpe_harmonic:
        y_h, y_p = librosa.effects.hpss(x)
        x_f0 = y_h.astype(np.float32, copy=False)
    else:
        x_f0 = x

    if rmvpe_bandpass:
        x_f0 = librosa.effects.preemphasis(x_f0)
        x_f0 = librosa.filters.get_window("hann", 1) * x_f0

    original_cwd = os.getcwd()
    os.chdir(sovits_rmvpe_dir)
    try:
        from RMVPEF0Predictor import RMVPEF0Predictor
        predictor = RMVPEF0Predictor(weight_path, device=device)
        predictor.threshold = float(uv_th)
        f0 = predictor.compute_f0(x_f0)
        f0 = np.asarray(f0, dtype=np.float32).reshape(-1)
    finally:
        os.chdir(original_cwd)

    if post:
        f0 = _soft_range_gate(f0, fmin, fmax, softness_hz=softness)

        voiced = _energy_gate(x, sr=sr, hop=160, frame=1024, thresh_db=gate_db)
        m = min(len(voiced), len(f0))
        f0 = f0[:m]
        voiced = voiced[:m]
        f0 = np.where(voiced, f0, 0.0).astype(np.float32)

        f0 = _fill_short_gaps(f0, sr=sr, hop=160, max_gap_ms=max_gap_ms)

    return _smooth_f0(f0, smooth)


def extract_crepe_pitch_and_periodicity(
    wav_path: str,
    uv_th: float = 0.000015,
    smooth: int = 1,
    device: str = "cpu",
    post: bool = False,
    gate_db: float = -42.0,
    fmin: float = 40.0,
    fmax: float = 1200.0,
    softness: float = 25.0,
    max_gap_ms: float = 60.0,
):
    """
    Returns:
      f0  : [T] 10ms grid
      prd : [T] 10ms grid
    Notes:
      - CREPE itself runs at 20ms (hop=320 here)
      - Both pitch and periodicity are repeated x2 -> 10ms grid
      - periodicity is saved as-is (not zeroed by uv threshold)
      - uv threshold is applied to pitch only
    """
    import librosa
    import numpy as np
    import torch
    import crepe

    x, sr = librosa.load(wav_path, sr=16000, mono=True)
    assert sr == 16000

    audio = torch.tensor(np.copy(x), dtype=torch.float32, device=device).unsqueeze(0)
    audio = audio + torch.randn_like(audio) * 0.001

    hop_length = 320  # 20ms
    base_fmin = 50
    base_fmax = 1000
    model = "full"
    batch_size = 512

    pitch_t, per_t = crepe.predict(
        audio,
        sr,
        hop_length,
        base_fmin,
        base_fmax,
        model,
        batch_size=batch_size,
        device=device,
        return_periodicity=True,
    )

    # same core processing as preprocessing side
    per_t = crepe.filter.median(per_t, 7)
    pitch_t = crepe.filter.mean(pitch_t, 5)

    # uv threshold only affects pitch
    pitch_t = torch.where(per_t >= float(uv_th), pitch_t, torch.zeros_like(pitch_t))

    pitch_np = pitch_t.squeeze(0).detach().cpu().numpy()
    per_np = per_t.squeeze(0).detach().cpu().numpy()

    # 20ms -> 10ms
    pitch_np = np.repeat(pitch_np, 2, axis=-1)
    per_np = np.repeat(per_np, 2, axis=-1)

    f0 = pitch_np.astype(np.float32).reshape(-1)
    prd = per_np.astype(np.float32).reshape(-1)

    f0 = np.where(np.isfinite(f0), f0, 0.0).astype(np.float32)
    prd = np.where(np.isfinite(prd), prd, 0.0).astype(np.float32)

    if post:
        f0 = _soft_range_gate(f0, fmin, fmax, softness_hz=softness)
        f0 = _fill_short_gaps(f0, sr=16000, hop=160, max_gap_ms=max_gap_ms)

    f0 = _smooth_f0(f0, smooth)
    prd = _smooth_scalar_track(prd, smooth)

    return f0, prd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-w", "--wav", required=True)
    p.add_argument("-p", "--pitch", required=True)  # 1列CSV
    p.add_argument("--method", default="rmvpe", choices=["rmvpe", "crepe"])
    p.add_argument("--uv-th", type=float, default=0.000015)
    p.add_argument("--smooth", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--post", action="store_true", help="apply post filters to f0 (rmvpe/crepe)")
    p.add_argument("--gate-db", type=float, default=-42.0, help="energy gate threshold in dB (higher = stricter)")
    p.add_argument("--fmin", type=float, default=40.0)
    p.add_argument("--fmax", type=float, default=1200.0)
    p.add_argument("--softness", type=float, default=25.0, help="soft range gate softness in Hz")
    p.add_argument("--max-gap-ms", type=float, default=60.0, help="fill short gaps (ms)")
    p.add_argument("--rmvpe-harmonic", action="store_true", help="use HPSS harmonic audio for RMVPE f0 extraction")
    p.add_argument("--rmvpe-bandpass", action="store_true", help="light band shaping for RMVPE f0 extraction (optional)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    if args.method == "rmvpe":
        # F0: RMVPE
        f0 = extract_f0_rmvpe(
            args.wav, args.uv_th, args.smooth, device,
            post=args.post, gate_db=args.gate_db, fmin=args.fmin, fmax=args.fmax,
            softness=args.softness, max_gap_ms=args.max_gap_ms,
            rmvpe_harmonic=args.rmvpe_harmonic,
            rmvpe_bandpass=args.rmvpe_bandpass,
        )
        # periodicity: CREPE only once here, so svc_inference.py never needs to call it
        _, prd = extract_crepe_pitch_and_periodicity(
            args.wav, args.uv_th, args.smooth, device,
            post=False, gate_db=args.gate_db,
            fmin=args.fmin, fmax=args.fmax,
            softness=args.softness, max_gap_ms=args.max_gap_ms
        )
    else:
        f0, prd = extract_crepe_pitch_and_periodicity(
            args.wav, args.uv_th, args.smooth, device,
            post=args.post, gate_db=args.gate_db,
            fmin=args.fmin, fmax=args.fmax,
            softness=args.softness, max_gap_ms=args.max_gap_ms
        )

    _write_csv_1col(args.pitch, f0)
    prd_path = _save_prd_from_pitch_csv_path(args.pitch, prd)

    print(args.pitch)
    print(prd_path)


if __name__ == "__main__":
    main()