<div align="center">
  <a href="./README_JP.md">日本語</a> |
  <a href="./README.md">English</a> |
  <a href="./README_ZH.md">简体中文</a>
</div>

# whisper-vits-svc Acoustic Dynamics Fork (声学动态分支版)

**实验性扩展分支**
基于 whisper-vits-svc，添加了修改以将演唱表现力（动态、呼吸感、爆发力、声音核心）显式重新注入模型中。

这个分支的目的并不仅仅是“改善音色”。
相反，它的目的是**将现有SVC在训练过程中容易平均化并丢弃的“演唱表现信息”，作为显式特征重新注入到模型中**。

如果您只需要原版的 whisper-vits-svc，请使用原仓库。此分支在显式提供动态特征方面具有强烈的主观意见（opinionated）。

---

## 使用方法（推理） — 推荐两阶段推理

在原仓库中，推理（基频解析和音频生成）可以一次性完成，但在实际运用（而非仅仅实验）中存在以下不便：
* 很难自由地使用 `--out` 指定文件名。
* 无法保存基频解析结果。每次更改后处理参数（如 `--uv-th` / `--max-gap-ms` / `--softness`）时，都必须重新运行繁重的 CREPE/RMVPE，以前的结果会被覆盖，这是一个痛点。
* 难以重用微调后的基频或将其应用于其他模型。

因此，此分支为了**注重实用性，更改为两阶段推理**：
*(虽然保留了省略 `--pit` 的单次运行模式，但如果想重用和调整基频，强烈推荐使用两阶段推理。)*

### UI（推荐）

```bash
python webui.py
```

浏览器会打开。选择工作文件夹，先提取音高，再转换。

* 音高 CSV 和 `.prd.npy` 会留在工作文件夹，方便反复调整 `--uv-th`
* 若有多个训练目录，可在转换页切换（config / 说话人从该目录读取）
* 请不要使用上游的 `app.py`（一次性推理）

命令行等价操作见下方。

### Stage 1: 基频 + 周期性 (Periodicity) 提取（繁重工作只需一次）

```bash
python pitch/inference.py \
  -w Unaccompanied.wav \
  -p Unaccompanied001.csv \
  --method crepe \          # 或 rmvpe
  --uv-th 0.0000000000015 \
  --max-gap-ms 3 \
  --post \                  # soft gate + gap fill 等后处理
  --smooth 1                # 注意：根据需要，这和 --post 可能是可选的。

```

* `output.pit.csv` (F0) 和 `output.prd.npy` (Periodicity) 将**自动保存在同一位置**。
* 对于呼吸成分较多的模型，我们确认使用极小的 `--uv-th` 值可以获得更好的结果。由于这取决于模型与输入音频的兼容性，因此很难设定一个通用推荐值。（有时 `--uv-th 0.015` 也能获得良好的结果）

### Stage 2: 实际转换（自动生成轻量级特征）

```bash
python svc_inference.py \
  --config configs/base.yaml \
  --model chkpt/NO-NAME/NO-NAME_0015.pt \
  --wave Unaccompanied.wav \
  --spk data_svc/singer/NO-NAME.spk.npy \
  --pit Unaccompanied001.csv \
  --out Unaccompanied001_NO-NAME.wav \
  --out-pit output_pitch.wav   # 可选：当推理结果出现奇怪的基频时，可用于调试确认

```

* `eng` / `deng` / `flat` 会**从输入的 wav 自动生成**（无需提前准备）。
* `prd` 会自动寻找 Stage 1 中创建的 `.prd.npy`。

### RMVPE 权重放置与路径问题（重要）

如果在推理时的基频解析中使用 RMVPE，请注意以下几点：

* 请根据需要将 RMVPE 模型放置在 `pretrain` 文件夹中。
* 代码中引用的是 `pretrain/model.pt`（也会使用 `sovits/rmvpe/` 目录）。
* 如果放置的是名为 `rmvpe.pt` 的文件，请在下载后重命名为 `model.pt`，或修改代码中的 `weight_path`。
* 分发示例：
* https://huggingface.co/RichardGR/so-vits-svc-pretrained/blob/main/rmvpe.pt
* https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip （解压后为 rmvpe.pt）


* 因为脚本内部使用的是相对路径解析，**在 Linux 和 Windows 下的行为可能会有所不同**。如果在您的环境中无法运行，请修改代码中的路径设置。

---

## 使用方法（训练）

### 数据集准备

移除了原仓库中的静音丢弃处理。请注意不要留下过长（30秒以上）的训练源音频。

在运行 `svc_preprocessing.py` 之前，您可以根据喜好使用附带的脚本（如 `prepare_length/build_global_29s.py`）。它会将原始数据按 RMS 分布或随机 best-of-n 均衡地组合成 29 秒以内的片段。
如果数据集过大导致脚本运行过于缓慢，请使用 `prepare_length/build_random_29s_for_big_dataset` 中的脚本（不分配 RMS，仅按 wav 长度进行随机连接）。

### 新增特征量的处理 (eng / deng / flat / prd)

* prepare 文件夹中添加了 `periodicity` / `log_energy` / `delta_energy` / `spectral_flatness` 的准备脚本。
* dataloader / train.py 侧已修改为将这些作为**强制输入**处理（没有关闭的标志）。
* 操作流程与原仓库相似，支持在 SVC 团队提供的 pretrain 模型上进行追加训练。

### 关于 base.yaml 特征量添加的注意事项

* `segment_size` 的设置变得更加严格。建议一开始缩小保存间隔，寻找适合该数据集的最佳参数点，然后再进行正式训练。
* 已确认提高 `accum_step` 会导致训练质量下降。

基本使用方法请参考原仓库。

---

## 背景与假设

现有的 whisper-vits-svc（以及类似的 SVC 模型）在音素、说话人身份和基频的再现方面表现优异。
然而，我们假设以下要素因为被隐式地嵌入在潜在表示中，所以在训练过程中很容易被平均化：

* 音量变化（动态）
* 发声的爆发力/张力
* 声带振动的稳定性
* 呼吸成分/噪声成分（非周期性）

这个分支的出发点是：**如果将这些信息作为显式特征提供给模型**，是否就能在转换后更好地保留“原始的演唱表现”？

### 试错结果（观察到的效果）

* 我们担心的“声音失真/暴走”现象，至少在我的数据集中并没有明显发生。
* 相反，我觉得**转换结果的表现力得到了提升**。
* 确认了一个现象：在呼吸成分较多的模型中，当极度降低 UV 阈值时，生成变得稳定且有所改善。
* **令人高兴的副作用**：以前在无音部分容易产生的“啊”的残留发声噪音大幅减少。在 DAW 上基本不再需要进行降噪处理了！（当然，这不能消除转换源自带的噪音）。

---

## 本分支添加的特征量（Acoustic Dynamics）

| 特征量 | 内容 | 提取来源 | 目的 |
| --- | --- | --- | --- |
| `log_energy` | 每帧的音量 (log RMS) | wav (RMS) | 整体动态 |
| `delta_energy` | 音量变化速度 (中心差分+3f MA) | 从 log_energy | 爆发力・重音 |
| `periodicity` | 周期性指标 (声音核心强度) | CREPE | 声带振动的稳定性 |
| `spectral_flatness` | 频谱平坦度 (噪声/呼吸近似) | wav 每帧的 FFT | 呼吸・摩擦音・噪声感 |

这些**不仅仅是简单的拼接输入 (concat)**，而是通过专用的 **Acoustic Dynamics Branch** 加算到 TextEncoder 内部。

```python
# models.py 摘录
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

这种结构在尽可能保持与原模型兼容性的同时，利用了附加信息。

---

## 训练管道相关的变更

除了引入追加特征量外，我们还对训练管道进行了重新审视。

* **顺序处理所有训练音频:** 在原仓库中，训练时并不一定处理完所有的训练音频。本分支更改了处理方式，改为顺序扫描训练数据。这使得掌握训练进度变得更加容易，也能清楚了解哪些音频已经被学习过。
* **更改为单 WAV 处理方式:**
不再同时加载多个 WAV，流程变为：“读取 1 个 WAV → 内部切分为片段 → 顺序训练”。根据环境的不同，这可能有助于降低 VRAM 的使用量。
* **以 WAV 为单位的进度管理:**
将 `epoch` 和 `epoch_wav_step` 分开管理。这使得可以每处理一定数量的音频就进行保存，并能更精细地确认训练进度（同时也保留了传统的 epoch 保存功能）。

---

## 注意事项・已知限制・免责声明

* 训练和推理都**仅在我个人的环境（Windows + 特定的数据集结构）下进行了充分验证**。重现性风险请自行承担。
* 代码中可能还残留着一些粗糙的部分。欢迎报告错误和提供改进建议，但我可能只能回答“在我的环境确认范围内是可以运行的”。

这项开发始于“能否保持转换前音频中包含的动态”以及“如果将呼吸成分作为特征量处理会怎样”的兴趣。AI 提出了特征量设计方案和代码，我通过选择、修改和验证推进了实现。
因为这是一个没有编程专业知识的人依靠 AI 搭建的的分支，肯定有许多我无法处理的地方。如果遇到“无法运行”的情况，向您身边的 AI 或更有知识的人请教可能是最快的捷径。

---

## Credits & Respect

对原仓库致以最深的感谢和敬意。

* **PlayVoice whisper-vits-svc:** [https://github.com/PlayVoice/whisper-vits-svc](https://github.com/PlayVoice/whisper-vits-svc)
* **so-vits-svc (RMVPE等):** [https://github.com/svc-develop-team/so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)

**見づ / Shidzuku**

---

# Original whisper-vits-svc Documentation

<div align="center">
<h1> Variational Inference with adversarial learning for end-to-end Singing Voice Conversion based on VITS </h1>
    
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/maxmax20160403/sovits5.0)
<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/PlayVoice/so-vits-svc-5.0">
<img alt="GitHub forks" src="https://img.shields.io/github/forks/PlayVoice/so-vits-svc-5.0">
<img alt="GitHub issues" src="https://img.shields.io/github/issues/PlayVoice/so-vits-svc-5.0">
<img alt="GitHub" src="https://img.shields.io/github/license/PlayVoice/so-vits-svc-5.0">

</div>

### 本项目使用简洁明了的代码结构，用于深度学习技术的研究
### 基于学习的目的，本项目并不追求效果极限、而更多的为学生笔记本考虑，采用了低配置参数、最终预训练模型为202M（包括生成器和判别器，且为float32模型），远远小于同类项目模型大小
### 如果你寻找的是直接可用的项目，本项目并不适合你

- 本项目的目标群体是：深度学习初学者，具备Python和PyTorch的基本操作是使用本项目的前置条件；
- 本项目旨在帮助深度学习初学者，摆脱枯燥的纯理论学习，通过与实践结合，熟练掌握深度学习基本知识；
- 本项目不支持实时变声；（支持需要换掉whisper）
- 本项目不会开发用于其他用途的一键包
### 代码详解课程
- 1-整体框架 https://www.bilibili.com/video/BV1Tj411e7pQ
- 2-数据准备和预处理 https://www.bilibili.com/video/BV1uj411v7zW
- 3-先验后验编码器 https://www.bilibili.com/video/BV1Be411Q7r5
- 4-decoder部分 https://www.bilibili.com/video/BV19u4y1b73U
- 5-蛇形激活函数 https://www.bilibili.com/video/BV1HN4y1D7AR
- 6-Flow部分 https://www.bilibili.com/video/BV1ju411F7Fs
- 7-训练及损失函数部分 https://www.bilibili.com/video/BV1qw411W73B
- 8-训练推理以及基频矫正 https://www.bilibili.com/video/BV1eb4y1u7ER

![vits-5.0-frame](https://github.com/PlayVoice/so-vits-svc-5.0/assets/16432329/3854b281-8f97-4016-875b-6eb663c92466)

- 【无 泄漏】支持多发音人

- 【捏 音色】创造独有发音人

- 【带 伴奏】也能进行转换，轻度伴奏

- 【用 Excel】进行原始调教，纯手工

https://github.com/PlayVoice/so-vits-svc-5.0/assets/16432329/63858332-cc0d-40e1-a216-6fe8bf638f7c

Powered by [@ShadowVap](https://space.bilibili.com/491283091)

## 模型特点：

| Feature | From | Status | Function |
| :--- | :--- | :--- | :--- |
| whisper | OpenAI | ✅ | 强大的抗噪能力 |
| bigvgan  | NVIDA | ✅ | 抗锯齿与蛇形激活，共振峰更清晰，提升音质明显 |
| natural speech | Microsoft | ✅ | 减少发音错误 |
| neural source-filter | NII | ✅ | 解决断音问题 |
| speaker encoder | Google | ✅ | 音色编码与聚类 |
| GRL for speaker | Ubisoft |✅ | 对抗去音色 |
| SNAC |  Samsung | ✅ | VITS 一句话克隆 |
| SCLN |  Microsoft | ✅ | 改善克隆 |
| PPG perturbation | 本项目 | ✅ | 提升抗噪性和去音色 |
| HuBERT perturbation | 本项目 | ✅ | 提升抗噪性和去音色 |
| VAE perturbation | 本项目 | ✅ | 提升音质 |
| Mix encoder | 本项目 | ✅ | 提升转换稳定性 |
| USP 推理 | 本项目 | ✅ | 提升转换稳定性 |

**USP : 即使unvoice和silence在推理的时候，也有Pitch，这个Pitch平滑链接voice段**
![vits_svc_usp](https://github.com/PlayVoice/so-vits-svc-5.0/assets/16432329/ba733b48-8a89-4612-83e0-a0745587d150)

## 为什么要mix

![mix_frame](https://github.com/PlayVoice/whisper-vits-svc/assets/16432329/3ffa1be0-1a21-4752-96b5-6220f98f2313)

## 安装环境

1. 安装[PyTorch](https://pytorch.org/get-started/locally/)

2.  安装项目依赖  
    ```
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
    ```
    **注意：不能额外安装whisper，否则会和代码内置whisper冲突**  

3.  下载[音色编码器](https://drive.google.com/drive/folders/15oeBYf6Qn1edONkVLXe82MzdIi3O_9m3), 把`best_model.pth.tar`放到`speaker_pretrain/`里面 （**不要解压**）

4.  下载[whisper-large-v2模型](https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt)，把`large-v2.pt`放到`whisper_pretrain/`里面

5.  下载[hubert_soft模型](https://github.com/bshall/hubert/releases/tag/v0.1)，把`hubert-soft-0d54a1f4.pt`放到`hubert_pretrain/`里面

6.  下载音高提取模型[crepe full](https://github.com/maxrmorrison/torchcrepe/tree/master/torchcrepe/assets)，把`full.pth`放到`crepe/assets`里面

    **注意：full.pth为84.9M，请确认文件大小无误**
    
7.  下载[sovits5.0.pretrain.pth](https://github.com/PlayVoice/so-vits-svc-5.0/releases/tag/5.0/), 把它放到`vits_pretrain/`里面，推理测试

    > python svc_inference.py --config configs/base.yaml --model ./vits_pretrain/sovits5.0.pretrain.pth --spk ./configs/singers/singer0001.npy --wave test.wav

## 数据集准备
1. 人声分离，如果数据集没有BGM直接跳过此步骤（推荐使用[UVR](https://github.com/Anjok07/ultimatevocalremovergui)中的3_HP-Vocal-UVR模型或者htdemucs_ft模型抠出数据集中的人声）  
2. 用[slicer](https://github.com/flutydeer/audio-slicer)剪切音频，whisper要求为小于30秒（建议丢弃不足2秒的音频，短音频大多没有音素，有可能会影响训练效果）  
3. 手动筛选经过第1步和第2步处理过的音频，裁剪或者丢弃杂音明显的音频，如果数据集没有BGM直接跳过此步骤  
4. 用Adobe Audition进行响度平衡处理  
5. 按下面文件结构，将数据集放入dataset_raw目录  
```shell
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

## 数据预处理

```shell
python svc_preprocessing.py -t 2
```
-t：指定线程数，必须是正整数且不得超过CPU总核心数，一般写2就可以了

预处理完成后文件夹结构如下面所示
```shell
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
    ├── speaker0.spk.npy
    └── speaker1.spk.npy
```

如果您有编程基础，推荐，逐步完成数据处理，也利于学习内部工作原理

- 1， 重采样

    生成采样率16000Hz音频, 存储路径为：./data_svc/waves-16k

    > python prepare/preprocess_a.py -w ./dataset_raw -o ./data_svc/waves-16k -s 16000

    生成采样率32000Hz音频, 存储路径为：./data_svc/waves-32k

    > python prepare/preprocess_a.py -w ./dataset_raw -o ./data_svc/waves-32k -s 32000

- 2， 使用16K音频，提取音高

    > python prepare/preprocess_crepe.py -w data_svc/waves-16k/ -p data_svc/pitch

- 3， 使用16k音频，提取内容编码
    > python prepare/preprocess_ppg.py -w data_svc/waves-16k/ -p data_svc/whisper

- 4， 使用16k音频，提取内容编码
    > python prepare/preprocess_hubert.py -w data_svc/waves-16k/ -v data_svc/hubert

- 5， 使用16k音频，提取音色编码
    > python prepare/preprocess_speaker.py data_svc/waves-16k/ data_svc/speaker

- 6， 提取音色编码均值；用于推理，也可作为发音人统一音色用于生成训练索引（数据音色变化不大的情况下）
    > python prepare/preprocess_speaker_ave.py data_svc/speaker/ data_svc/singer

- 7， 使用32k音频，提取线性谱
    > python prepare/preprocess_spec.py -w data_svc/waves-32k/ -s data_svc/specs

- 8， 使用32k音频，生成训练索引
    > python prepare/preprocess_train.py

- 9， 训练文件调试
    > python prepare/preprocess_zzz.py

## 训练
0. 参数调整  
  如果基于预训练模型微调，需要下载预训练模型[sovits5.0.pretrain.pth](https://github.com/PlayVoice/so-vits-svc-5.0/releases/tag/5.0)并且放在项目根目录下面<br>
  并且修改`configs/base.yaml`的参数`pretrain: "./vits_pretrain/sovits5.0.pretrain.pth"`，并适当调小学习率（建议从5e-5开始尝试）<br>
  **learning_rate & batch_size & accum_step 为三个紧密相关的参数，需要仔细调节**<br>
  **batch_size 乘以 accum_step 通常等于 16 或 32，对于低显存GPU，可以尝试 batch_size = 4，accum_step = 4**

1. 开始训练  
   ```
   python svc_trainer.py -c configs/base.yaml -n sovits5.0
   ```
2. 恢复训练
   ```
   python svc_trainer.py -c configs/base.yaml -n sovits5.0 -p chkpt/sovits5.0/sovits5.0_***.pt
   ```
3. 训练日志可视化
   ```
   tensorboard --logdir logs/
   ```

![sovits5 0_base](https://github.com/PlayVoice/so-vits-svc-5.0/assets/16432329/1628e775-5888-4eac-b173-a28dca978faa)

![sovits_spec](https://github.com/PlayVoice/so-vits-svc-5.0/assets/16432329/c4223cf3-b4a0-4325-bec0-6d46d195a1fc)

## 推理
1. 导出推理模型：文本编码器，Flow网络，Decoder网络；判别器和后验编码器等只在训练中使用  
   ```
   python svc_export.py --config configs/base.yaml --checkpoint_path chkpt/sovits5.0/***.pt
   ```
2. 推理  
- 如果不想手动调整f0，只需要最终的推理结果，运行下面的命令即可
  ```
  python svc_inference.py --config configs/base.yaml --model sovits5.0.pth --spk ./data_svc/singer/修改成对应的名称.npy --wave test.wav --shift 0
  ```
- 如果需要手动调整f0，依据下面的流程操作

  - 使用whisper提取内容编码，生成test.ppg.npy
    ```
    python whisper/inference.py -w test.wav -p test.ppg.npy
    ```

  - 使用hubert提取内容编码，生成test.vec.npy
    ```
    python hubert/inference.py -w test.wav -v test.vec.npy
    ```

  - 提取csv文本格式F0参数，用Excel打开csv文件，对照Audition或者SonicVisualiser手动修改错误的F0
     ```
     python pitch/inference.py -w test.wav -p test.csv
     ```
  - 最终推理
     ```
     python svc_inference.py --config configs/base.yaml --model sovits5.0.pth --spk ./data_svc/singer/修改成对应的名称.npy --wave test.wav --ppg test.ppg.npy --vec test.vec.npy --pit test.csv --shift 0
     ```

3. 一些注意点  
    当指定--ppg后，多次推理同一个音频时，可以避免重复提取音频内容编码；没有指定，也会自动提取  
    
    当指定--vec后，多次推理同一个音频时，可以避免重复提取音频内容编码；没有指定，也会自动提取  
    
    当指定--pit后，可以加载手工调教的F0参数；没有指定，也会自动提取  
    
    生成文件在当前目录svc_out.wav
    
    | args | --config | --model | --spk | --wave | --ppg | --vec | --pit | --shift |
    | :---:  | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
    | name | 配置文件 | 模型文件 | 音色文件 | 音频文件 | ppg内容 | hubert内容 | 音高内容 | 升降调 |

4. 去噪后处理
```
python svc_inference_post.py --ref test.wav --svc svc_out.wav --out svc_out_post.wav
```

## 两种训练模式
- 分散模式：训练索引中，音色文件使用音频音色
- 统一模式：训练索引中，音色文件使用发音人音色

**问题：哪种情况下，哪个模式更好**

## 模型融合
```
python svc_merge.py --model1 模型1.pt --model1 模型2.pt --rate 模型1占比(0~1)
```
对不同epoch的模型进行融合，可以获得比较平均的性能、削弱过拟合

例如：python svc_merge.py --model1 chkpt\sovits5.0\sovits5.0_1045.pt --model2 chkpt\sovits5.0\sovits5.0_1050.pt --rate 0.4

## 捏音色
纯属巧合的取名：average -> ave -> eva，夏娃代表者孕育和繁衍
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

生成的音色文件为：eva.spk.npy

## 数据集

| Name | URL |
| :--- | :--- |
|KiSing         |http://shijt.site/index.php/2021/05/16/kising-the-first-open-source-mandarin-singing-voice-synthesis-corpus/|
|PopCS          |https://github.com/MoonInTheRiver/DiffSinger/blob/master/resources/apply_form.md|
|opencpop       |https://wenet.org.cn/opencpop/download/|
|Multi-Singer   |https://github.com/Multi-Singer/Multi-Singer.github.io|
|M4Singer       |https://github.com/M4Singer/M4Singer/blob/master/apply_form.md|
|CSD            |https://zenodo.org/record/4785016#.YxqrTbaOMU4|
|KSS            |https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset|
|JVS MuSic      |https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_music|
|PJS            |https://sites.google.com/site/shinnosuketakamichi/research-topics/pjs_corpus|
|JUST Song      |https://sites.google.com/site/shinnosuketakamichi/publication/jsut-song|
|MUSDB18        |https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems|
|DSD100         |https://sigsep.github.io/datasets/dsd100.html|
|Aishell-3      |http://www.aishelltech.com/aishell_3|
|VCTK           |https://datashare.ed.ac.uk/handle/10283/2651|
|Korean Songs   |http://urisori.co.kr/urisori-en/doku.php/|

## 代码来源和参考文献

https://github.com/facebookresearch/speech-resynthesis [paper](https://arxiv.org/abs/2104.00355)

https://github.com/jaywalnut310/vits [paper](https://arxiv.org/abs/2106.06103)

https://github.com/openai/whisper/ [paper](https://arxiv.org/abs/2212.04356)

https://github.com/NVIDIA/BigVGAN [paper](https://arxiv.org/abs/2206.04658)

https://github.com/mindslab-ai/univnet [paper](https://arxiv.org/abs/2106.07889)

https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/tree/master/project/01-nsf

https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS

https://github.com/brentspell/hifi-gan-bwe

https://github.com/mozilla/TTS

https://github.com/bshall/soft-vc

https://github.com/maxrmorrison/torchcrepe

https://github.com/MoonInTheRiver/DiffSinger

https://github.com/OlaWod/FreeVC [paper](https://arxiv.org/abs/2210.15418)

https://github.com/yl4579/HiFTNet [paper](https://arxiv.org/abs/2309.09493)

[One-shot Voice Conversion by Separating Speaker and Content Representations with Instance Normalization](https://arxiv.org/abs/1904.05742)

[SNAC : Speaker-normalized Affine Coupling Layer in Flow-based Architecture for Zero-Shot Multi-Speaker Text-to-Speech](https://github.com/hcy71o/SNAC)

[Adapter-Based Extension of Multi-Speaker Text-to-Speech Model for New Speakers](https://arxiv.org/abs/2211.00585)

[AdaSpeech: Adaptive Text to Speech for Custom Voice](https://arxiv.org/pdf/2103.00993.pdf)

[AdaVITS: Tiny VITS for Low Computing Resource Speaker Adaptation](https://arxiv.org/pdf/2206.00208.pdf)

[Cross-Speaker Prosody Transfer on Any Text for Expressive Speech Synthesis](https://github.com/ubisoft/ubisoft-laforge-daft-exprt)

[Learn to Sing by Listening: Building Controllable Virtual Singer by Unsupervised Learning from Voice Recordings](https://arxiv.org/abs/2305.05401)

[Adversarial Speaker Disentanglement Using Unannotated External Data for Self-supervised Representation Based Voice Conversion](https://arxiv.org/pdf/2305.09167.pdf)

[Multilingual Speech Synthesis and Cross-Language Voice Cloning: GRL](https://arxiv.org/abs/1907.04448)

[RoFormer: Enhanced Transformer with rotary position embedding](https://arxiv.org/abs/2104.09864))https://github.com/facebookresearch/speech-resynthesis [paper](https://arxiv.org/abs/2104.00355)

## 基于数据扰动防止音色泄露的方法

https://github.com/auspicious3000/contentvec/blob/main/contentvec/data/audio/audio_utils_1.py

https://github.com/revsic/torch-nansy/blob/main/utils/augment/praat.py

https://github.com/revsic/torch-nansy/blob/main/utils/augment/peq.py

https://github.com/biggytruck/SpeechSplit2/blob/main/utils.py

https://github.com/OlaWod/FreeVC/blob/main/preprocess_sr.py

## 贡献者

<a href="https://github.com/PlayVoice/so-vits-svc/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PlayVoice/so-vits-svc" />
</a>

## 特别感谢

https://github.com/Francis-Komizu/Sovits

## 原创过程
2022.04.12 https://mp.weixin.qq.com/s/autNBYCsG4_SvWt2-Ll_zA

2022.04.22 https://github.com/PlayVoice/VI-SVS

2022.07.26 https://mp.weixin.qq.com/s/qC4TJy-4EVdbpvK2cQb1TA

2022.09.08 https://github.com/PlayVoice/VI-SVC

## 被这个项目拷贝：svc-develop-team/so-vits-svc
![coarse_f0_1](https://github.com/PlayVoice/so-vits-svc-5.0/assets/16432329/e2f5e5d3-d169-42c1-953f-4e1648b6da37)

![coarse_f0_2](https://github.com/PlayVoice/so-vits-svc-5.0/assets/16432329/f3539c83-7c8a-425e-bf20-2c402132f0f4)

![coarse_f0_3](https://github.com/PlayVoice/so-vits-svc-5.0/assets/16432329/f3cee94a-0eeb-4189-b9bb-7043d06e62ef)

## Rcell对拷贝的真实回应

![Rcell](https://github.com/PlayVoice/so-vits-svc-5.0/assets/16432329/8ebb236d-e233-4cea-9359-8e44029b5af5)
