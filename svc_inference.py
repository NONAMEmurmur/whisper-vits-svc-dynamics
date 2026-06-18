import logging
import sys, os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import argparse
import numpy as np
import librosa

from omegaconf import OmegaConf
from scipy.io.wavfile import write
from vits.models import SynthesizerInfer
from pitch import load_csv_pitch
from feature_retrieval import IRetrieval, DummyRetrieval, FaissIndexRetrieval, load_retrieve_index

logger = logging.getLogger(__name__)

EPS_ENERGY = 1e-5
EPS_FLAT = 1e-8


def ensure_parent_dir(path_str: str):
    path = Path(path_str)
    if path.parent != Path(""):
        path.parent.mkdir(parents=True, exist_ok=True)


def get_speaker_name_from_path(speaker_path: Path) -> str:
    suffixes = "".join(speaker_path.suffixes)
    filename = speaker_path.name
    return filename.rstrip(suffixes)


def create_retrival(cli_args) -> IRetrieval:
    if not cli_args.enable_retrieval:
        logger.info("infer without retrival")
        return DummyRetrieval()

    logger.info("load index retrival model")

    speaker_name = get_speaker_name_from_path(Path(cli_args.spk))
    base_path = Path(".").absolute() / "data_svc" / "indexes" / speaker_name

    if cli_args.hubert_index_path:
        hubert_index_filepath = cli_args.hubert_index_path
    else:
        index_name = f"{cli_args.retrieval_index_prefix}hubert.index"
        hubert_index_filepath = base_path / index_name

    if cli_args.whisper_index_path:
        whisper_index_filepath = cli_args.whisper_index_path
    else:
        index_name = f"{cli_args.retrieval_index_prefix}whisper.index"
        whisper_index_filepath = base_path / index_name

    return FaissIndexRetrieval(
        hubert_index=load_retrieve_index(
            filepath=hubert_index_filepath,
            ratio=cli_args.retrieval_ratio,
            n_nearest_vectors=cli_args.n_retrieval_vectors
        ),
        whisper_index=load_retrieve_index(
            filepath=whisper_index_filepath,
            ratio=cli_args.retrieval_ratio,
            n_nearest_vectors=cli_args.n_retrieval_vectors
        ),
    )


def load_svc_model(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_state_dict = checkpoint_dict["model_g"]
    state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except Exception:
            print("%s is not in the checkpoint" % k)
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    return model


def infer_prd_path_from_pitch_csv(pit_csv_path: str) -> str:
    if pit_csv_path.endswith(".csv"):
        return pit_csv_path[:-4] + ".prd.npy"
    return pit_csv_path + ".prd.npy"


def moving_average(x: np.ndarray, k: int = 3) -> np.ndarray:
    if k <= 1:
        return x.astype(np.float32)
    pad = k // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(x_pad, kernel, mode="valid").astype(np.float32)


def extract_frames_32k(y: np.ndarray, frame_length: int = 1024, hop_length: int = 320) -> np.ndarray:
    """
    32k wav -> 10ms grid (hop=320)
    Tail is zero-padded so final partial frame is also kept.
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
        frames.append(y[start:start + frame_length])

    return np.asarray(frames, dtype=np.float32)


def generate_energy_flatness_from_wav(
    wav_path: str,
    sr: int = 32000,
    frame_length: int = 1024,
    hop_length: int = 320,
):
    """
    Returns:
      eng  : log_energy [T]
      deng : delta_energy [T]
      flat : spectral_flatness [T]
    """
    y, loaded_sr = librosa.load(wav_path, sr=sr, mono=True)
    assert loaded_sr == sr

    frames = extract_frames_32k(y, frame_length=frame_length, hop_length=hop_length)

    if len(frames) == 0:
        eng = np.zeros((0,), dtype=np.float32)
        deng = np.zeros((0,), dtype=np.float32)
        flat = np.zeros((0,), dtype=np.float32)
        return eng, deng, flat

    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    eng = np.log(rms + EPS_ENERGY).astype(np.float32)

    deng = np.zeros_like(eng, dtype=np.float32)
    if len(eng) >= 3:
        deng[1:-1] = 0.5 * (eng[2:] - eng[:-2])
        deng[0] = eng[1] - eng[0]
        deng[-1] = eng[-1] - eng[-2]
    elif len(eng) == 2:
        deng[0] = eng[1] - eng[0]
        deng[1] = eng[1] - eng[0]
    else:
        deng[0] = 0.0
    deng = moving_average(deng, 3)

    window = np.hanning(frame_length).astype(np.float32)
    windowed = frames * window[None, :]
    spec = np.abs(np.fft.rfft(windowed, axis=1)).astype(np.float32)
    spec = np.maximum(spec, EPS_FLAT)

    geometric_mean = np.exp(np.mean(np.log(spec), axis=1))
    arithmetic_mean = np.mean(spec, axis=1)
    flat = (geometric_mean / np.maximum(arithmetic_mean, EPS_FLAT)).astype(np.float32)

    return eng, deng, flat


def maybe_prepare_extra_features(args):
    """
    Returns numpy arrays:
      eng, deng, prd, flat
    prd is always loaded from pit-related .prd.npy
    eng/deng/flat are loaded from files if all provided, otherwise generated from wav
    """
    prd_path = infer_prd_path_from_pitch_csv(args.pit)
    if not os.path.isfile(prd_path):
        raise FileNotFoundError(
            f".prd.npy not found: {prd_path}\n"
            "Generate pitch with pitch/inference.py first so periodicity is saved."
        )
    prd = np.load(prd_path).astype(np.float32)

    if args.eng is not None and args.deng is not None and args.flat is not None:
        eng = np.load(args.eng).astype(np.float32)
        deng = np.load(args.deng).astype(np.float32)
        flat = np.load(args.flat).astype(np.float32)
    else:
        eng, deng, flat = generate_energy_flatness_from_wav(args.wave)

    return eng, deng, prd, flat


def svc_infer(model, retrieval, spk, pit, ppg, vec, eng, deng, prd, flat, hp, device, out_pit_path):
    len_min = min(len(pit), len(vec), len(ppg), len(eng), len(deng), len(prd), len(flat))

    pit = pit[:len_min]
    vec = vec[:len_min, :]
    ppg = ppg[:len_min, :]
    eng = eng[:len_min]
    deng = deng[:len_min]
    prd = prd[:len_min]
    flat = flat[:len_min]

    with torch.no_grad():
        spk = spk.unsqueeze(0).to(device)

        source = pit.unsqueeze(0).to(device)
        source = model.pitch2source(source)
        pitwav = model.source2wav(source)

        ensure_parent_dir(out_pit_path)
        write(out_pit_path, hp.data.sampling_rate, pitwav)

        hop_size = hp.data.hop_length
        all_frame = len_min
        hop_frame = 10
        out_chunk = 2500
        out_index = 0
        out_audio = []

        while out_index < all_frame:
            cut_s = 0 if out_index == 0 else out_index - hop_frame
            cut_s_out = 0 if out_index == 0 else hop_frame * hop_size

            if out_index + out_chunk + hop_frame > all_frame:
                cut_e = all_frame
                cut_e_out = -1
            else:
                cut_e = out_index + out_chunk + hop_frame
                cut_e_out = -hop_frame * hop_size

            sub_ppg = retrieval.retriv_whisper(ppg[cut_s:cut_e, :])
            sub_vec = retrieval.retriv_hubert(vec[cut_s:cut_e, :])

            sub_ppg = sub_ppg.unsqueeze(0).to(device)
            sub_vec = sub_vec.unsqueeze(0).to(device)
            sub_pit = pit[cut_s:cut_e].unsqueeze(0).to(device)
            sub_eng = eng[cut_s:cut_e].unsqueeze(0).to(device)
            sub_deng = deng[cut_s:cut_e].unsqueeze(0).to(device)
            sub_prd = prd[cut_s:cut_e].unsqueeze(0).to(device)
            sub_flat = flat[cut_s:cut_e].unsqueeze(0).to(device)

            sub_len = torch.LongTensor([cut_e - cut_s]).to(device)
            sub_har = source[:, :, cut_s * hop_size:cut_e * hop_size].to(device)

            # NOTE:
            # current modified models.py signature:
            # inference(ppg, vec, pit, spk, ppg_l, source, eng, deng, prd, flat)
            sub_out = model.inference(
                sub_ppg,
                sub_vec,
                sub_pit,
                spk,
                sub_len,
                sub_har,
                sub_eng,
                sub_deng,
                sub_prd,
                sub_flat,
            )[0, 0].cpu().numpy()

            sub_out = sub_out[cut_s_out:cut_e_out]
            out_audio.extend(sub_out)
            out_index += out_chunk

    return np.asarray(out_audio)


def main(args):
    if args.ppg is None:
        args.ppg = "svc_tmp.ppg.npy"
        os.system(f"python whisper/inference.py -w {args.wave} -p {args.ppg}")

    if args.vec is None:
        args.vec = "svc_tmp.vec.npy"
        os.system(f"python hubert/inference.py -w {args.wave} -v {args.vec}")

    if args.pit is None:
        args.pit = "svc_tmp.pit.csv"
        os.system(f"python pitch/inference.py -w {args.wave} -p {args.pit} --method {args.pitch_method}")

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hp = OmegaConf.load(args.config)

    model = SynthesizerInfer(
        hp.data.filter_length // 2 + 1,
        hp.data.segment_size // hp.data.hop_length,
        hp
    )
    load_svc_model(args.model, model)

    retrieval = create_retrival(args)

    model.eval().to(device)

    spk = torch.FloatTensor(np.load(args.spk))

    ppg = torch.FloatTensor(np.repeat(np.load(args.ppg), 2, 0))
    vec = torch.FloatTensor(np.repeat(np.load(args.vec), 2, 0))
    pit = torch.FloatTensor(load_csv_pitch(args.pit))

    eng_np, deng_np, prd_np, flat_np = maybe_prepare_extra_features(args)

    eng = torch.FloatTensor(eng_np)
    deng = torch.FloatTensor(deng_np)
    prd = torch.FloatTensor(prd_np)
    flat = torch.FloatTensor(flat_np)

    if args.shift != 0:
        pit = pit * (2 ** (args.shift / 12))

    out_audio = svc_infer(
        model, retrieval, spk, pit, ppg, vec, eng, deng, prd, flat, hp, device, args.out_pit
    )

    ensure_parent_dir(args.out)
    write(args.out, hp.data.sampling_rate, out_audio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--wave', type=str, required=True)
    parser.add_argument('--spk', type=str, required=True)
    parser.add_argument('--ppg', type=str)
    parser.add_argument('--vec', type=str)
    parser.add_argument('--pit', type=str)
    parser.add_argument('--shift', type=int, default=0)

    # optional precomputed extra features
    parser.add_argument('--eng', type=str)
    parser.add_argument('--deng', type=str)
    parser.add_argument('--flat', type=str)

    # only used when --pit is omitted and pitch/inference.py is auto-called
    parser.add_argument('--pitch-method', type=str, default='rmvpe', choices=['rmvpe', 'crepe'])

    parser.add_argument('--out', type=str, default="svc_out.wav")
    parser.add_argument('--out-pit', dest='out_pit', type=str, default="svc_out_pit.wav")

    parser.add_argument('--enable-retrieval', action="store_true")
    parser.add_argument('--retrieval-index-prefix', default='')
    parser.add_argument('--retrieval-ratio', type=float, default=.5)
    parser.add_argument('--n-retrieval-vectors', type=int, default=3)
    parser.add_argument('--hubert-index-path')
    parser.add_argument('--whisper-index-path')

    parser.add_argument('--debug', action="store_true")

    args = parser.parse_args()
    main(args)