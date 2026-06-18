import os
import argparse
import numpy as np
import librosa
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


EPS = 1e-5


def moving_average(x, k=3):
    if k <= 1:
        return x.astype(np.float32)
    pad = k // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(x_pad, kernel, mode="valid").astype(np.float32)


def extract_frames(y, frame_length=1024, hop_length=320):
    """
    10ms grid on 32k audio:
      hop_length = 320 samples
    We generate frames by stepping every 320 samples.
    Tail is zero-padded so that the final partial frame is also kept.
    """
    n = len(y)
    if n == 0:
        return np.zeros((0, frame_length), dtype=np.float32)

    num_frames = int(np.ceil(n / hop_length))
    total_length = (num_frames - 1) * hop_length + frame_length
    if total_length > n:
        y = np.pad(y, (0, total_length - n), mode="constant")

    frames = []
    for i in range(num_frames):
        start = i * hop_length
        frame = y[start:start + frame_length]
        frames.append(frame)

    return np.asarray(frames, dtype=np.float32)


def compute_energy(wav_path, eng_save_path, deng_save_path, sr=32000, frame_length=1024, hop_length=320):
    y, loaded_sr = librosa.load(wav_path, sr=sr)
    assert loaded_sr == sr

    frames = extract_frames(y, frame_length=frame_length, hop_length=hop_length)

    if len(frames) == 0:
        log_energy = np.zeros((0,), dtype=np.float32)
        delta = np.zeros((0,), dtype=np.float32)
    else:
        rms = np.sqrt(np.mean(np.square(frames), axis=1))
        log_energy = np.log(rms + EPS).astype(np.float32)

        delta = np.zeros_like(log_energy, dtype=np.float32)
        if len(log_energy) >= 3:
            delta[1:-1] = 0.5 * (log_energy[2:] - log_energy[:-2])
            delta[0] = log_energy[1] - log_energy[0]
            delta[-1] = log_energy[-1] - log_energy[-2]
        elif len(log_energy) == 2:
            delta[0] = log_energy[1] - log_energy[0]
            delta[1] = log_energy[1] - log_energy[0]
        else:
            delta[0] = 0.0

        delta = moving_average(delta, 3)

    np.save(eng_save_path, log_energy.astype(np.float32), allow_pickle=False)
    np.save(deng_save_path, delta.astype(np.float32), allow_pickle=False)


def process_file(file, wav_root, out_root, spk, sr, frame_length, hop_length):
    if not file.endswith(".wav"):
        return

    stem = file[:-4]
    wav_path = f"{wav_root}/{spk}/{file}"
    eng_save_path = f"{out_root}/{spk}/{stem}.eng"
    deng_save_path = f"{out_root}/{spk}/{stem}.deng"

    compute_energy(
        wav_path,
        eng_save_path,
        deng_save_path,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
    )


def process_speaker(wav_root, out_root, spk, sr, frame_length, hop_length, thread_num):
    files = [f for f in os.listdir(f"./{wav_root}/{spk}") if f.endswith(".wav")]

    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        futures = {
            executor.submit(
                process_file,
                file,
                wav_root,
                out_root,
                spk,
                sr,
                frame_length,
                hop_length,
            ): file
            for file in files
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing energy {spk}"):
            future.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wav", help="input 32k wav root", dest="wav", required=True)
    parser.add_argument("-e", "--eng", help="output energy root", dest="eng", required=True)
    parser.add_argument("-s", "--sr", help="sample rate", dest="sr", type=int, default=32000)
    parser.add_argument("--frame", help="frame length", dest="frame_length", type=int, default=1024)
    parser.add_argument("--hop", help="hop length", dest="hop_length", type=int, default=320)
    parser.add_argument(
        "-t", "--thread_count",
        help="thread count to process, set 0 to use all cpu cores",
        dest="thread_count",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    print(args.wav)
    print(args.eng)
    print(args.sr)
    print(args.frame_length)
    print(args.hop_length)

    assert args.sr == 32000
    assert args.hop_length == 320

    os.makedirs(args.eng, exist_ok=True)

    wav_root = args.wav
    out_root = args.eng

    for spk in os.listdir(wav_root):
        if os.path.isdir(f"./{wav_root}/{spk}"):
            os.makedirs(f"./{out_root}/{spk}", exist_ok=True)

            if args.thread_count == 0:
                process_num = os.cpu_count() // 2 + 1
            else:
                process_num = args.thread_count

            process_speaker(
                wav_root,
                out_root,
                spk,
                args.sr,
                args.frame_length,
                args.hop_length,
                process_num,
            )