<div align="center">
  <a href="./README_JP.md">日本語</a> |
  <a href="./README.md">English</a> |
  <a href="./README_ZH.md">简体中文</a>
</div>

# whisper-vits-svc Acoustic Dynamics Fork

**Experimental Extension Fork**
Based on whisper-vits-svc, this fork introduces modifications to explicitly re-inject singing expressions (dynamics, breathiness, vocal push, and vocal core) into the model.

The goal of this fork is **not** simply to "improve voice quality".
Rather, it aims to **explicitly re-inject "performance information"—which existing SVC models tend to average out and discard during the training process—as explicit acoustic features**.

If you want the vanilla whisper-vits-svc, please use the original repository. This fork is heavily opinionated towards explicitly providing dynamics.

---

## Usage (Inference) - Two-Stage Inference Recommended

In the original repository, inference (pitch extraction and audio generation) could be done in a single step. However, for practical use rather than just experimentation, this was inconvenient:
* It is difficult to freely specify the filename with `--out`.
* Pitch analysis results cannot be saved. If you want to test different post-processing parameters (`--uv-th`, `--max-gap-ms`, `--softness`), you are forced to re-run the heavy CREPE/RMVPE models every time, overwriting previous results.
* It is difficult to reuse fine-tuned pitch or apply it to other models.

Therefore, this fork heavily modifies the workflow to a **practical two-stage inference**:
*(The one-shot mode omitting `--pit` is still available, but two-stage inference is strongly recommended for pitch reuse and adjustment.)*

### UI (recommended)

```bash
python webui.py
```

A browser window opens. Pick a working folder, extract pitch, then convert.

* Pitch CSV and `.prd.npy` stay in the working folder, so you can iterate on `--uv-th`
* If you keep multiple training folders, switch them in the Convert tab (config / speakers are loaded from that folder)
* Do not use the upstream `app.py`; it is one-shot inference

The CLI equivalent is below.

### Stage 1: Pitch + Periodicity Extraction (Run heavy tasks only once)

```bash
python pitch/inference.py \
  -w Unaccompanied.wav \
  -p Unaccompanied001.csv \
  --method crepe \          # or rmvpe
  --uv-th 0.0000000000015 \
  --max-gap-ms 3 \
  --post \                  # Post-processing (soft gate + gap fill, etc.)
  --smooth 1                # Note: --smooth and --post might be optional depending on your needs.

```

* `output.pit.csv` (F0) and `output.prd.npy` (Periodicity) are **automatically saved in the same location**.
* For models with a lot of breathy components, we have confirmed that using an extremely small `--uv-th` value yields better results. Since it depends on the compatibility between the model and the input audio, it's difficult to set a universal recommended value. (Sometimes `--uv-th 0.015` works just fine).

### Stage 2: Actual Conversion (Lightweight features are auto-generated)

```bash
python svc_inference.py \
  --config configs/base.yaml \
  --model chkpt/NO-NAME/NO-NAME_0015.pt \
  --wave Unaccompanied.wav \
  --spk data_svc/singer/NO-NAME.spk.npy \
  --pit Unaccompanied001.csv \
  --out Unaccompanied001_NO-NAME.wav \
  --out-pit output_pitch.wav   # Optional: useful for debugging weird pitch artifacts

```

* `eng`, `deng`, and `flat` are **automatically generated from the input wav** (no preparation needed).
* `prd` automatically looks for the `.prd.npy` created in Stage 1.

### RMVPE Weights Placement and Path Issues (Important)

If you use RMVPE for pitch extraction during inference, please note:

* Place the RMVPE model in the `pretrain` folder if necessary.
* The code references `pretrain/model.pt` (it also uses the `sovits/rmvpe/` directory).
* If you downloaded a file named `rmvpe.pt`, rename it to `model.pt` or modify the `weight_path` in the code.
* Distribution examples:
* https://huggingface.co/RichardGR/so-vits-svc-pretrained/blob/main/rmvpe.pt
* https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip (Extract `rmvpe.pt`)


* Because relative paths are resolved inside the script, **behavior may differ between Linux and Windows**. If it doesn't work in your environment, adjust the path settings in the code.

---

## Usage (Training)

### Dataset Preparation

The silent drop processing from the original repository has been removed. Be careful not to leave training sources that are too long (over 30 seconds).

Before running `svc_preprocessing.py`, you can optionally use the provided scripts (e.g., `prepare_length/build_global_29s.py`) to balance the raw data into segments under 29 seconds using RMS distribution or random best-of-n.
If your dataset is too massive and the RMS calculation takes too long, use the script in `prepare_length/build_random_29s_for_big_dataset` (random concatenation by wav-length only, without RMS distribution).

### Handling the newly added features (eng / deng / flat / prd)

* Preparation scripts for `periodicity`, `log_energy`, `delta_energy`, and `spectral_flatness` have been added to the `prepare` directory.
* `dataloader` and `train.py` have been modified to treat these as **mandatory inputs** (there is no flag to turn them off).
* The process is designed to allow further training on the pretrain models provided by the SVC team, just like the original repository.

### Notice regarding `base.yaml` and added features

* `segment_size` settings have become more strict. We recommend starting with a shorter save interval to find the sweet spot for your dataset before running a full training session.
* It has been confirmed that increasing `accum_step` degrades training quality.

For basic usage, please refer to the original repository.

---

## Background and Hypothesis

Existing whisper-vits-svc (and similar SVC models) excel at reproducing phonemes, speaker identity, and pitch.
However, we hypothesized that the following elements tend to be averaged out during training because they are implicitly embedded in latent representations:

* Volume changes (Dynamics)
* Vocal push / tension
* Stability of vocal fold vibration
* Breath / Noise components (Aperiodicity)

The starting point of this fork was the idea that **giving these pieces of information to the model as explicit features** would help retain the "raw singing expression" even after conversion.

### Trial and Error Results (Observed Effects)

* The anticipated "audio artifacts/instability" did not occur noticeably (at least with my datasets).
* Instead, I feel the **expressiveness of the conversion results has improved**.
* Confirmed a phenomenon where generation stabilizes/improves when lowering the UV threshold extremely on models with heavy breath components.
* **Happy side effect:** The residual vocal noise (a faint "Ah" sound) that used to occur during silent parts has significantly decreased. Noise reduction in a DAW is essentially no longer necessary! (Of course, noise originating from the source audio cannot be erased).

---

## Added Features (Acoustic Dynamics)

| Feature | Content | Extracted from | Purpose |
| --- | --- | --- | --- |
| `log_energy` | Volume per frame (log RMS) | wav (RMS) | Overall dynamics |
| `delta_energy` | Speed of volume change (Central diff + 3f MA) | log_energy | Vocal push / Accents |
| `periodicity` | Periodicity index (Vocal core strength) | CREPE | Stability of vocal fold vibration |
| `spectral_flatness` | Spectral flatness (Noise/Breath approx.) | wav frame FFT | Breathiness / Aperiodicity |

These are **not mere concatenated inputs**. They are added to the TextEncoder internally via a dedicated **Acoustic Dynamics Branch**.

```python
# excerpt from models.py
extra_proj = nn.Sequential(
    nn.Conv1d(4, 64, 1),
    nn.SiLU(),
    nn.Conv1d(64, hidden_channels, 1),
)
...
if eng is not None and deng is not None and prd is not None and flat is not None:
    extra = self.extra_proj(torch.stack([eng, deng, prd, flat], dim=1)) * x_mask
    x = x + extra

```

This structure utilizes the additional information while maintaining as much compatibility with the original model as possible.

---

## Changes to the Training Pipeline

In addition to adding features, the training pipeline has been overhauled.

* **Sequential processing of all training audio:** The original repo did not necessarily process all training audio during an epoch. This fork changes the processing method to sequentially scan the training data, making it easier to track progress and know which audio has been learned.
* **Shift to single WAV processing:**
Instead of loading multiple WAVs simultaneously, the flow is now: "Read 1 WAV → Split into segments internally → Train sequentially." This can potentially reduce VRAM usage depending on the environment.
* **Progress management per WAV:**
`epoch` and `epoch_wav_step` are now managed separately. This allows saving checkpoints per a certain number of audio processed, and finer confirmation of training progress (traditional epoch saving is also retained).

---

## Notes, Known Limitations, and Disclaimer

* Both training and inference have **only been sufficiently tested in my personal environment (Windows + specific dataset configuration)**. Reproducibility is at your own risk.
* There may be rough parts left in the code. Bug reports and improvements are welcome, but I might only be able to answer, "It worked within the scope I confirmed in my environment."

This development started from the curiosity of "Can we preserve the dynamics in the pre-conversion audio?" and "What happens if we treat breath components as features?". The AI proposed feature designs and code, and I proceeded with the implementation by selecting, modifying, and verifying them.
Because this fork was built by someone without professional coding knowledge relying on AI, there must be many parts I cannot handle. If you encounter an "it doesn't work" situation, consulting your local AI or someone with more knowledge might be the fastest shortcut.

---

## Credits & Respect

Huge thanks and respect to the original repositories:

* **PlayVoice whisper-vits-svc:** [https://github.com/PlayVoice/whisper-vits-svc](https://github.com/PlayVoice/whisper-vits-svc)
* **so-vits-svc (RMVPE, etc.):** [https://github.com/svc-develop-team/so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)

**見づ / Shidzuku**

---

# Original whisper-vits-svc Documentation

The tree [bigvgan-mix-v2](https://www.google.com/search?q=https://github.com/PlayVoice/whisper-vits-svc/tree/bigvgan-mix-v2) has good audio quality

The tree [RoFormer-HiFTNet](https://www.google.com/search?q=https://github.com/PlayVoice/whisper-vits-svc/tree/RoFormer-HiFTNet) has fast infer speed

No More Upgrade

* This project targets deep learning beginners, basic knowledge of Python and PyTorch are the prerequisites for this project;
* This project aims to help deep learning beginners get rid of boring pure theoretical learning, and master the basic knowledge of deep learning by combining it with practices;
* This project does not support real-time voice converting; (need to replace whisper if real-time voice converting is what you are looking for)
* This project will not develop one-click packages for other purposes;

* A minimum VRAM requirement of 6GB for training
* Support for multiple speakers
* Create unique speakers through speaker mixing
* It can even convert voices with light accompaniment
* You can edit F0 using Excel

https://github.com/PlayVoice/so-vits-svc-5.0/assets/16432329/6a09805e-ab93-47fe-9a14-9cbc1e0e7c3a

Powered by [@ShadowVap](https://www.google.com/search?q=https://space.bilibili.com/491283091)

## Model properties

| Feature | From | Status | Function |
| --- | --- | --- | --- |
| whisper | OpenAI | ✅ | strong noise immunity |
| bigvgan | NVIDA | ✅ | alias and snake |
| natural speech | Microsoft | ✅ | reduce mispronunciation |
| neural source-filter | Xin Wang | ✅ | solve the problem of audio F0 discontinuity |
| pitch quantization | Xin Wang | ✅ | quantize the F0 for embedding |
| speaker encoder | Google | ✅ | Timbre Encoding and Clustering |
| GRL for speaker | Ubisoft | ✅ | Preventing Encoder Leakage Timbre |
| SNAC | Samsung | ✅ | One Shot Clone of VITS |
| SCLN | Microsoft | ✅ | Improve Clone |
| Diffusion | HuaWei | ✅ | Improve sound quality |
| PPG perturbation | this project | ✅ | Improved noise immunity and de-timbre |
| HuBERT perturbation | this project | ✅ | Improved noise immunity and de-timbre |
| VAE perturbation | this project | ✅ | Improve sound quality |
| MIX encoder | this project | ✅ | Improve conversion stability |
| USP infer | this project | ✅ | Improve conversion stability |
| HiFTNet | Columbia University | ✅ | NSF-iSTFTNet for speed up |
| RoFormer | Zhuiyi Technology | ✅ | Rotary Positional Embeddings |

due to the use of data perturbation, it takes longer to train than other projects.

**USP : Unvoice and Silence with Pitch when infer**


## Why mix

## Plug-In-Diffusion

## Setup Environment

1. Install [PyTorch](https://www.google.com/search?q=https://pytorch.org/get-started/locally/).
2. Install project dependencies

```shell
    pip install -i [https://pypi.tuna.tsinghua.edu.cn/simple](https://pypi.tuna.tsinghua.edu.cn/simple) -r requirements.txt
    ```
**Note: whisper is already built-in, do not install it again otherwise it will cuase conflict and error**
3. Download the Timbre Encoder: [Speaker-Encoder by @mueller91](https://drive.google.com/drive/folders/15oeBYf6Qn1edONkVLXe82MzdIi3O_9m3), put `best_model.pth.tar`  into `speaker_pretrain/`.

4. Download whisper model [whisper-large-v2](https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt). Make sure to download `large-v2.pt`，put it into `whisper_pretrain/`.

5. Download [hubert_soft model](https://github.com/bshall/hubert/releases/tag/v0.1)，put `hubert-soft-0d54a1f4.pt` into `hubert_pretrain/`.

6. Download pitch extractor [crepe full](https://github.com/maxrmorrison/torchcrepe/tree/master/torchcrepe/assets)，put `full.pth` into `crepe/assets`.

**Note: crepe full.pth is 84.9 MB, not 6kb**

7. Download pretrain model [sovits5.0.pretrain.pth](https://github.com/PlayVoice/so-vits-svc-5.0/releases/tag/5.0/), and put it into `vits_pretrain/`.
```shell
    python svc_inference.py --config configs/base.yaml --model ./vits_pretrain/sovits5.0.pretrain.pth --spk ./configs/singers/singer0001.npy --wave test.wav
    ```

## Dataset preparation

Necessary pre-processing:
1. Separate voice and accompaniment with [UVR](https://github.com/Anjok07/ultimatevocalremovergui) (skip if no accompaniment)
2. Cut audio input to shorter length with [slicer](https://github.com/flutydeer/audio-slicer), whisper takes input less than 30 seconds.
3. Manually check generated audio input, remove inputs shorter than 2 seconds or with obivous noise.
4. Adjust loudness if necessary, recommend Adobe Audiiton.
5. Put the dataset into the `dataset_raw` directory following the structure below.

```

dataset_raw
├───speaker0
│   ├───000001.wav
│   ├───...
│   └───000xxx.wav
└───speaker1
├───000001.wav
├───...
└───000xxx.wav

```

## Data preprocessing
```shell
python svc_preprocessing.py -t 2

```

`-t`: threading, max number should not exceed CPU core count, usually 2 is enough.
After preprocessing you will get an output with following structure.

```
data_svc/
└── waves-16k
│    └── speaker0
│    │      ├── 000001.wav
│    │      └── 000xxx.wav
│    └── speaker1
│           ├── 000001.wav
│           └── 000xxx.wav
└── waves-32k
│    └── speaker0
│    │      ├── 000001.wav
│    │      └── 000xxx.wav
│    └── speaker1
│           ├── 000001.wav
│           └── 000xxx.wav
└── pitch
│    └── speaker0
│    │      ├── 000001.pit.npy
│    │      └── 000xxx.pit.npy
│    └── speaker1
│           ├── 000001.pit.npy
│           └── 000xxx.pit.npy
└── hubert
│    └── speaker0
│    │      ├── 000001.vec.npy
│    │      └── 000xxx.vec.npy
│    └── speaker1
│           ├── 000001.vec.npy
│           └── 000xxx.vec.npy
└── whisper
│    └── speaker0
│    │      ├── 000001.ppg.npy
│    │      └── 000xxx.ppg.npy
│    └── speaker1
│           ├── 000001.ppg.npy
│           └── 000xxx.ppg.npy
└── speaker
│    └── speaker0
│    │      ├── 000001.spk.npy
│    │      └── 000xxx.spk.npy
│    └── speaker1
│           ├── 000001.spk.npy
│           └── 000xxx.spk.npy
└── singer
│   ├── speaker0.spk.npy
│   └── speaker1.spk.npy
|
└── indexes
    ├── speaker0
    │   ├── some_prefix_hubert.index
    │   └── some_prefix_whisper.index
    └── speaker1
        ├── hubert.index
        └── whisper.index

```

1. Re-sampling
* Generate audio with a sampling rate of 16000Hz in `./data_svc/waves-16k`



```
    python prepare/preprocess_a.py -w ./dataset_raw -o ./data_svc/waves-16k -s 16000
    ```

- Generate audio with a sampling rate of 32000Hz in `./data_svc/waves-32k`

```

```
python prepare/preprocess_a.py -w ./dataset_raw -o ./data_svc/waves-32k -s 32000
```

```

2. Use 16K audio to extract pitch

```
    python prepare/preprocess_crepe.py -w data_svc/waves-16k/ -p data_svc/pitch
    ```
3. Use 16K audio to extract ppg

```

```
python prepare/preprocess_ppg.py -w data_svc/waves-16k/ -p data_svc/whisper
```

```

4. Use 16K audio to extract hubert

```
    python prepare/preprocess_hubert.py -w data_svc/waves-16k/ -v data_svc/hubert
    ```
5. Use 16k audio to extract timbre code

```

```
python prepare/preprocess_speaker.py data_svc/waves-16k/ data_svc/speaker
```

```

6. Extract the average value of the timbre code for inference; it can also replace a single audio timbre in generating the training index, and use it as the unified timbre of the speaker for training

```
    python prepare/preprocess_speaker_ave.py data_svc/speaker/ data_svc/singer
    ``` 
7. Use 32k audio to extract the linear spectrum

```

```
python prepare/preprocess_spec.py -w data_svc/waves-32k/ -s data_svc/specs
``` 

```

8. Use 32k audio to generate training index

```
    python prepare/preprocess_train.py
    ```
11. Training file debugging

```

```
python prepare/preprocess_zzz.py
```

```

## Train

1. If fine-tuning is based on the pre-trained model, you need to download the pre-trained model: [sovits5.0.pretrain.pth](https://www.google.com/search?q=https://github.com/PlayVoice/so-vits-svc-5.0/releases/tag/5.0). Put pretrained model under project root, change this line

```
    pretrain: "./vits_pretrain/sovits5.0.pretrain.pth"
    ```
in `configs/base.yaml`，and adjust the learning rate appropriately, eg 5e-5.

`batch_size`: for GPU with 6G VRAM, 6 is the recommended value, 8 will work but step speed will be much slower.
2. Start training

```

python svc_trainer.py -c configs/base.yaml -n sovits5.0

```
3. Resume training

```

python svc_trainer.py -c configs/base.yaml -n sovits5.0 -p chkpt/sovits5.0/sovits5.0_***.pt

```
4. Log visualization

```

tensorboard --logdir logs/

```

![sovits5 0_base](https://github.com/PlayVoice/so-vits-svc-5.0/assets/16432329/1628e775-5888-4eac-b173-a28dca978faa)

![sovits_spec](https://github.com/PlayVoice/so-vits-svc-5.0/assets/16432329/c4223cf3-b4a0-4325-bec0-6d46d195a1fc)

## Inference

1. Export inference model: text encoder, Flow network, Decoder network

```

python svc_export.py --config configs/base.yaml --checkpoint_path chkpt/sovits5.0/***.pt

```
2. Inference
- if there is no need to adjust `f0`, just run the following command.

```

python svc_inference.py --config configs/base.yaml --model sovits5.0.pth --spk ./data_svc/singer/your_singer.spk.npy --wave test.wav --shift 0

```
- if `f0` will be adjusted manually, follow the steps:
1. use whisper to extract content encoding, generate `test.vec.npy`.

```

```
   python whisper/inference.py -w test.wav -p test.ppg.npy
   ```

```

2. use hubert to extract content vector, without using one-click reasoning, in order to reduce GPU memory usage

```
       python hubert/inference.py -w test.wav -v test.vec.npy
       ```
3. extract the F0 parameter to the csv text format, open the csv file in Excel, and manually modify the wrong F0 according to Audition or SonicVisualiser

```

```
   python pitch/inference.py -w test.wav -p test.csv
   ```

```

4. final inference

```
       python svc_inference.py --config configs/base.yaml --model sovits5.0.pth --spk ./data_svc/singer/your_singer.spk.npy --wave test.wav --ppg test.ppg.npy --vec test.vec.npy --pit test.csv --shift 0
       ```
3. Notes

- when `--ppg` is specified, when the same audio is reasoned multiple times, it can avoid repeated extraction of audio content codes; if it is not specified, it will be automatically extracted;

- when `--vec` is specified, when the same audio is reasoned multiple times, it can avoid repeated extraction of audio content codes; if it is not specified, it will be automatically extracted;

- when `--pit` is specified, the manually tuned F0 parameter can be loaded; if not specified, it will be automatically extracted;

- generate files in the current directory:svc_out.wav

4. Arguments ref

| args |--config | --model | --spk | --wave | --ppg | --vec | --pit | --shift |
| :---:  | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| name | config path | model path | speaker | wave input | wave ppg | wave hubert | wave pitch | pitch shift |

5. post by vad

```

python svc_inference_post.py --ref test.wav --svc svc_out.wav --out svc_out_post.wav

```

## Train Feature Retrieval Index (Optional)

To increase the stability of the generated timbre, you can use the method described in the 
[Retrieval-based-Voice-Conversion](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/en/README.en.md) 
repository. This method consists of 2 steps: 

1. Training the retrieval index on hubert and whisper features
    Run training with default settings:

```

```
python svc_train_retrieval.py
```

```

If the number of vectors is more than 200_000 they will be compressed to 10_000 using the MiniBatchKMeans algorithm.
You can change these settings using command line options:

```
    usage: crate faiss indexes for feature retrieval [-h] [--debug] [--prefix PREFIX] [--speakers SPEAKERS [SPEAKERS ...]] [--compress-features-after COMPRESS_FEATURES_AFTER]
                                                     [--n-clusters N_CLUSTERS] [--n-parallel N_PARALLEL]

    options:
      -h, --help            show this help message and exit
      --debug
      --prefix PREFIX       add prefix to index filename
      --speakers SPEAKERS [SPEAKERS ...]
                            speaker names to create an index. By default all speakers are from data_svc
      --compress-features-after COMPRESS_FEATURES_AFTER
                            If the number of features is greater than the value compress feature vectors using MiniBatchKMeans.
      --n-clusters N_CLUSTERS
                            Number of centroids to which features will be compressed
      --n-parallel N_PARALLEL
                            Nuber of parallel job of MinibatchKmeans. Default is cpus-1
    ``` 
Compression of training vectors can speed up index inference, but reduces the quality of the retrieve.
Use vector count compression if you really have a lot of them.

The resulting indexes will be stored in the "indexes" folder as:

```

```
data_svc
...
└── indexes
    ├── speaker0
    │   ├── some_prefix_hubert.index
    │   └── some_prefix_whisper.index
    └── speaker1
        ├── hubert.index
        └── whisper.index
```

```

2. At the inference stage adding the n closest features in a certain proportion of the vits model
Enable Feature Retrieval with settings:

```
    python svc_inference.py --config configs/base.yaml --model sovits5.0.pth --spk ./data_svc/singer/your_singer.spk.npy --wave test.wav --shift 0 \
    --enable-retrieval \
    --retrieval-ratio 0.5 \
    --n-retrieval-vectors 3
    ``` 
For a better retrieval effect, you can try to cycle through different parameters: `--retrieval-ratio` and `--n-retrieval-vectors`

If you have multiple sets of indexes, you can specify a specific set via the parameter: `--retrieval-index-prefix`

You can explicitly specify the paths to the hubert and whisper indexes using the parameters: `--hubert-index-path` and `--whisper-index-path`


## Create singer
named by pure coincidence：average -> ave -> eva，eve(eva) represents conception and reproduction


```

python svc_eva.py

```

```python
eva_conf = {
    './configs/singers/singer0022.npy': 0,
    './configs/singers/singer0030.npy': 0,
    './configs/singers/singer0047.npy': 0.5,
    './configs/singers/singer0051.npy': 0.5,
}

```

the generated singer file will be `eva.spk.npy`.

## Data set

| Name | URL |
| --- | --- |
| KiSing | http://shijt.site/index.php/2021/05/16/kising-the-first-open-source-mandarin-singing-voice-synthesis-corpus/ |
| PopCS | https://github.com/MoonInTheRiver/DiffSinger/blob/master/resources/apply_form.md |
| opencpop | https://wenet.org.cn/opencpop/download/ |
| Multi-Singer | https://github.com/Multi-Singer/Multi-Singer.github.io |
| M4Singer | https://github.com/M4Singer/M4Singer/blob/master/apply_form.md |
| CSD | https://zenodo.org/record/4785016#.YxqrTbaOMU4 |
| KSS | https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset |
| JVS MuSic | https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_music |
| PJS | https://sites.google.com/site/shinnosuketakamichi/research-topics/pjs_corpus |
| JUST Song | https://sites.google.com/site/shinnosuketakamichi/publication/jsut-song |
| MUSDB18 | https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems |
| DSD100 | https://sigsep.github.io/datasets/dsd100.html |
| Aishell-3 | http://www.aishelltech.com/aishell_3 |
| VCTK | https://datashare.ed.ac.uk/handle/10283/2651 |
| Korean Songs | http://urisori.co.kr/urisori-en/doku.php/ |

## Code sources and references

https://github.com/facebookresearch/speech-resynthesis [paper](https://www.google.com/search?q=https%3A%2F%2Farxiv.org%2Fabs%2F2104.00355)

https://github.com/jaywalnut310/vits [paper](https://www.google.com/search?q=https%3A%2F%2Farxiv.org%2Fabs%2F2106.06103)

https://github.com/openai/whisper/ [paper](https://www.google.com/search?q=https%3A%2F%2Farxiv.org%2Fabs%2F2212.04356)

https://github.com/NVIDIA/BigVGAN [paper](https://www.google.com/search?q=https%3A%2F%2Farxiv.org%2Fabs%2F2206.04658)

https://github.com/mindslab-ai/univnet [paper](https://www.google.com/search?q=https%3A%2F%2Farxiv.org%2Fabs%2F2106.07889)

https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/tree/master/project/01-nsf

https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS

https://github.com/brentspell/hifi-gan-bwe

https://github.com/mozilla/TTS

https://github.com/bshall/soft-vc

https://github.com/maxrmorrison/torchcrepe

https://github.com/MoonInTheRiver/DiffSinger

https://github.com/OlaWod/FreeVC [paper](https://www.google.com/search?q=https%3A%2F%2Farxiv.org%2Fabs%2F2210.15418)

https://github.com/yl4579/HiFTNet [paper](https://www.google.com/search?q=https%3A%2F%2Farxiv.org%2Fabs%2F2309.09493)

[Autoregressive neural f0 model for statistical parametric speech synthesis](https://www.google.com/search?q=https%3A%2F%2Fweb.archive.org%2Fweb%2F20210718024752id_%2Fhttps%3A%2F%2Fieeexplore.ieee.org%2Fielx7%2F6570655%2F8356719%2F08341752.pdf)

[One-shot Voice Conversion by Separating Speaker and Content Representations with Instance Normalization](https://www.google.com/search?q=https%3A%2F%2Farxiv.org%2Fabs%2F1904.05742)

[SNAC : Speaker-normalized Affine Coupling Layer in Flow-based Architecture for Zero-Shot Multi-Speaker Text-to-Speech](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fhcy71o%2FSNAC)

[Adapter-Based Extension of Multi-Speaker Text-to-Speech Model for New Speakers](https://www.google.com/search?q=https%3A%2F%2Farxiv.org%2Fabs%2F2211.00585)

[AdaSpeech: Adaptive Text to Speech for Custom Voice](https://www.google.com/search?q=https%3A%2F%2Farxiv.org%2Fpdf%2F2103.00993.pdf)

[AdaVITS: Tiny VITS for Low Computing Resource Speaker Adaptation](https://www.google.com/search?q=https%3A%2F%2Farxiv.org%2Fpdf%2F2206.00208.pdf)

[Cross-Speaker Prosody Transfer on Any Text for Expressive Speech Synthesis](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fubisoft%2Fubisoft-laforge-daft-exprt)

[Learn to Sing by Listening: Building Controllable Virtual Singer by Unsupervised Learning from Voice Recordings](https://www.google.com/search?q=https%3A%2F%2Farxiv.org%2Fabs%2F2305.05401)

[Adversarial Speaker Disentanglement Using Unannotated External Data for Self-supervised Representation Based Voice Conversion](https://www.google.com/search?q=https%3A%2F%2Farxiv.org%2Fpdf%2F2305.09167.pdf)

[Multilingual Speech Synthesis and Cross-Language Voice Cloning: GRL](https://www.google.com/search?q=https%3A%2F%2Farxiv.org%2Fabs%2F1907.04448)

[RoFormer: Enhanced Transformer with rotary position embedding](https://www.google.com/search?q=https%3A%2F%2Farxiv.org%2Fabs%2F2104.09864)

## Method of Preventing Timbre Leakage Based on Data Perturbation

https://github.com/auspicious3000/contentvec/blob/main/contentvec/data/audio/audio_utils_1.py

https://github.com/revsic/torch-nansy/blob/main/utils/augment/praat.py

https://github.com/revsic/torch-nansy/blob/main/utils/augment/peq.py

https://github.com/biggytruck/SpeechSplit2/blob/main/utils.py

https://github.com/OlaWod/FreeVC/blob/main/preprocess_sr.py

## Contributors

## Thanks to

https://github.com/Francis-Komizu/Sovits

## Relevant Projects

* [LoRA-SVC](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2FPlayVoice%2Flora-svc): decoder only svc
* [Grad-SVC](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2FPlayVoice%2FGrad-SVC): diffusion based svc

## Original evidence

2022.04.12 https://mp.weixin.qq.com/s/autNBYCsG4_SvWt2-Ll_zA

2022.04.22 https://github.com/PlayVoice/VI-SVS

2022.07.26 https://mp.weixin.qq.com/s/qC4TJy-4EVdbpvK2cQb1TA

2022.09.08 https://github.com/PlayVoice/VI-SVC

## Be copied by svc-develop-team/so-vits-svc