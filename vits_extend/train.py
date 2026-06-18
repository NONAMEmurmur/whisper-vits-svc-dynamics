import os
import time
import logging
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel

from vits_extend.dataloader import create_dataloader_train
from vits_extend.dataloader import create_dataloader_eval
from vits_extend.writer import MyWriter
from vits_extend.stft import TacotronSTFT
from vits_extend.stft_loss import MultiResolutionSTFTLoss
from vits_extend.validation import validate
from vits_decoder.discriminator import Discriminator
from vits.models import SynthesizerTrn
from vits import commons
from vits.losses import kl_loss
from vits.commons import clip_grad_value_


def load_model(model, saved_state_dict, tag="model", logger=None):
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    new_state_dict = {}
    loaded = []
    missing = []
    shape_mismatch = []

    for k, v in state_dict.items():
        if k not in saved_state_dict:
            new_state_dict[k] = v
            missing.append(k)
            continue

        if saved_state_dict[k].shape != v.shape:
            new_state_dict[k] = v
            shape_mismatch.append((k, tuple(saved_state_dict[k].shape), tuple(v.shape)))
            continue

        new_state_dict[k] = saved_state_dict[k]
        loaded.append(k)

    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)

    summary = f"[LOAD {tag}] loaded={len(loaded)} missing={len(missing)} shape_mismatch={len(shape_mismatch)}"
    if logger is not None:
        logger.info(summary)
        for k in missing:
            logger.warning(f"[MISSING {tag}] {k}")
        for k, src_shape, dst_shape in shape_mismatch:
            logger.warning(f"[SHAPE_MISMATCH {tag}] {k}: ckpt{src_shape} != model{dst_shape}")
    else:
        print(summary)
    return model


def resolve_resume_safe_save_path(pth_dir, run_name, epoch, chkpt_path):
    base_path = os.path.join(pth_dir, f"{run_name}_{epoch:04d}.pt")
    if chkpt_path is None:
        return base_path

    try:
        resume_abs = os.path.abspath(chkpt_path)
        target_abs = os.path.abspath(base_path)
    except Exception:
        return base_path

    if resume_abs != target_abs:
        return base_path

    i = 1
    while True:
        candidate = os.path.join(pth_dir, f"{run_name}_{epoch:04d}_resume{i}.pt")
        try:
            candidate_abs = os.path.abspath(candidate)
        except Exception:
            return candidate

        if candidate_abs != resume_abs and not os.path.exists(candidate):
            return candidate
        i += 1


def split_one_wav_to_segments(ppg, ppg_l, vec, pit, spk, spec, spec_l, audio, audio_l, eng, deng, prd, flat, hp):
    segment_frames = hp.data.segment_size // hp.data.hop_length
    usable_frames = min(
        int(ppg_l.item()),
        int(spec_l.item()),
        int(audio_l.item()) // hp.data.hop_length,
        int(vec.size(1)),
        int(pit.size(1)),
        int(eng.size(1)),
        int(deng.size(1)),
        int(prd.size(1)),
        int(flat.size(1)),
    )
    nseg = usable_frames // segment_frames
    out = []
    for idx in range(nseg):
        fs = idx * segment_frames
        fe = fs + segment_frames
        ws = fs * hp.data.hop_length
        we = fe * hp.data.hop_length
        out.append({
            "ppg": ppg[:, fs:fe, :],
            "ppg_l": torch.LongTensor([segment_frames]).to(ppg.device),
            "vec": vec[:, fs:fe, :],
            "pit": pit[:, fs:fe],
            "spk": spk,
            "spec": spec[:, :, fs:fe],
            "spec_l": torch.LongTensor([segment_frames]).to(spec.device),
            "audio": audio[:, :, ws:we],
            "audio_l": torch.LongTensor([hp.data.segment_size]).to(audio.device),
            "eng": eng[:, fs:fe],
            "deng": deng[:, fs:fe],
            "prd": prd[:, fs:fe],
            "flat": flat[:, fs:fe],
        })
    return out


def run_one_segment(seg, model_g, model_d, stft, stft_criterion, spkc_criterion, hp, device):
    ppg = seg["ppg"]
    ppg_l = seg["ppg_l"]
    vec = seg["vec"]
    pit = seg["pit"]
    spk = seg["spk"]
    spec = seg["spec"]
    spec_l = seg["spec_l"]
    audio = seg["audio"]
    eng = seg["eng"]
    deng = seg["deng"]
    prd = seg["prd"]
    flat = seg["flat"]

    fake_audio, ids_slice, z_mask, \
        (z_f, z_r, z_p, m_p, logs_p, z_q, m_q, logs_q, logdet_f, logdet_r), spk_preds = model_g(
            ppg, vec, pit, spec, spk, ppg_l, spec_l, eng, deng, prd, flat
        )

    audio = commons.slice_segments(audio, ids_slice * hp.data.hop_length, hp.data.segment_size)

    spk_loss = spkc_criterion(
        spk,
        spk_preds,
        torch.Tensor(spk_preds.size(0)).to(device).fill_(1.0)
    )

    mel_fake = stft.mel_spectrogram(fake_audio.squeeze(1))
    mel_real = stft.mel_spectrogram(audio.squeeze(1))
    mel_loss = F.l1_loss(mel_fake, mel_real) * hp.train.c_mel

    sc_loss, mag_loss = stft_criterion(fake_audio.squeeze(1), audio.squeeze(1))
    stft_loss = (sc_loss + mag_loss) * hp.train.c_stft

    disc_fake = model_d(fake_audio)
    score_loss = 0.0
    for (_, score_fake) in disc_fake:
        score_loss += torch.mean(torch.pow(score_fake - 1.0, 2))
    score_loss = score_loss / len(disc_fake)

    disc_real = model_d(audio)
    feat_loss = 0.0
    for (feat_fake, _), (feat_real, _) in zip(disc_fake, disc_real):
        for fake, real in zip(feat_fake, feat_real):
            feat_loss += torch.mean(torch.abs(fake - real))
    feat_loss = feat_loss / len(disc_fake)
    feat_loss = feat_loss * 2

    loss_kl_f = kl_loss(z_f, logs_q, m_p, logs_p, logdet_f, z_mask) * hp.train.c_kl
    loss_kl_r = kl_loss(z_r, logs_p, m_q, logs_q, logdet_r, z_mask) * hp.train.c_kl

    loss_g = score_loss + feat_loss + mel_loss + stft_loss + loss_kl_f + loss_kl_r * 0.5 + spk_loss * 2
    loss_d = None

    stats = {
        "loss_g_item": float(loss_g.item()),
        "loss_s_item": float(stft_loss.item()),
        "loss_m_item": float(mel_loss.item()),
        "loss_k_item": float(loss_kl_f.item()),
        "loss_r_item": float(loss_kl_r.item()),
        "loss_i_item": float(spk_loss.item()),
        "score_item": float(score_loss.item()),
    }

    return fake_audio, audio, loss_g, stats


def run_discriminator(fake_audio, audio, model_d):
    disc_fake = model_d(fake_audio.detach())
    disc_real = model_d(audio)

    loss_d = 0.0
    for (_, score_fake), (_, score_real) in zip(disc_fake, disc_real):
        loss_d += torch.mean(torch.pow(score_real - 1.0, 2))
        loss_d += torch.mean(torch.pow(score_fake, 2))
    loss_d = loss_d / len(disc_fake)
    return loss_d


def save_checkpoint(model_g, model_d, optim_g, optim_d, args, hp_str, step, epoch, path,
                    epoch_wav_step=0, val_index=0):
    torch.save({
        'model_g': (model_g.module if args.num_gpus > 1 else model_g).state_dict(),
        'model_d': (model_d.module if args.num_gpus > 1 else model_d).state_dict(),
        'optim_g': optim_g.state_dict(),
        'optim_d': optim_d.state_dict(),
        'step': step,
        'epoch': epoch,
        'epoch_wav_step': epoch_wav_step,
        'val_index': val_index,
        'hp_str': hp_str,
    }, path)


def save_and_validate(rank, logger, model_g, model_d, optim_g, optim_d, args, hp_str,
                      step, epoch, save_path, hp, valloader, stft, writer, device,
                      val_index, epoch_wav_step):
    if rank != 0:
        return val_index

    save_checkpoint(
        model_g, model_d, optim_g, optim_d, args, hp_str, step, epoch, save_path,
        epoch_wav_step=epoch_wav_step, val_index=val_index
    )
    logger.info("Saved checkpoint to: %s" % save_path)
    logger.info(
        "Running validation | epoch=%d step=%d epoch_wav_step=%d val_index=%d lr_g=%.10f lr_d=%.10f" % (
            epoch, step, epoch_wav_step, val_index,
            optim_g.param_groups[0]['lr'], optim_d.param_groups[0]['lr']
        )
    )
    with torch.no_grad():
        validate(
            hp, args, model_g, model_d, valloader, stft, writer,
            step, device, epoch, val_index=val_index
        )
    return val_index + 1


def clean_checkpoints(path_to_models, run_name, keep_ckpts, logger):
    if keep_ckpts <= 0:
        return

    ckpts_files = [
        f for f in os.listdir(path_to_models)
        if os.path.isfile(os.path.join(path_to_models, f))
    ]

    def is_target(fn):
        return fn.startswith(run_name) and fn.endswith(".pt")

    x_sorted = sorted(
        [f for f in ckpts_files if is_target(f)],
        key=lambda _f: os.path.getmtime(os.path.join(path_to_models, _f))
    )

    to_del = [os.path.join(path_to_models, fn) for fn in x_sorted[:-keep_ckpts]]
    for fn in to_del:
        logger.info(f"Free up space by deleting ckpt {fn}")
        os.remove(fn)


def train(rank, args, chkpt_path, hp, hp_str):
    if args.num_gpus > 1:
        init_process_group(
            backend=hp.dist_config.dist_backend,
            init_method=hp.dist_config.dist_url,
            world_size=hp.dist_config.world_size * args.num_gpus,
            rank=rank
        )

    torch.cuda.manual_seed(hp.train.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    model_g = SynthesizerTrn(
        hp.data.filter_length // 2 + 1,
        hp.data.segment_size // hp.data.hop_length,
        hp
    ).to(device)
    model_d = Discriminator(hp).to(device)

    optim_g = torch.optim.AdamW(
        model_g.parameters(),
        lr=hp.train.learning_rate,
        betas=hp.train.betas,
        eps=hp.train.eps
    )
    optim_d = torch.optim.AdamW(
        model_d.parameters(),
        lr=(hp.train.learning_rate / hp.train.accum_step),
        betas=hp.train.betas,
        eps=hp.train.eps
    )

    init_epoch = 1
    step = 0
    val_index = 0
    epoch_wav_step_resume = 0
    resumed_mid_epoch = False

    stft = TacotronSTFT(
        filter_length=hp.data.filter_length,
        hop_length=hp.data.hop_length,
        win_length=hp.data.win_length,
        n_mel_channels=hp.data.mel_channels,
        sampling_rate=hp.data.sampling_rate,
        mel_fmin=hp.data.mel_fmin,
        mel_fmax=hp.data.mel_fmax,
        center=False,
        device=device
    )

    logger = None
    writer = None
    valloader = None
    pth_dir = None
    log_dir = None

    if rank == 0:
        pth_dir = os.path.join(hp.log.pth_dir, args.name)
        log_dir = os.path.join(hp.log.log_dir, args.name)
        os.makedirs(pth_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, '%s-%d.log' % (args.name, time.time()))),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger()
        writer = MyWriter(hp, log_dir)
        valloader = create_dataloader_eval(hp)
        logger.info("[LR_MODE] using base.yaml learning_rate as-is on fresh start")
        logger.info("[LR_MODE] base learning_rate = %.10f" % hp.train.learning_rate)

    run_head_validate = False
    run_head_validate_epoch = 0
    run_head_validate_reason = "initial"

    if chkpt_path is not None:
        if rank == 0:
            logger.info("Resuming from checkpoint: %s" % chkpt_path)

        checkpoint = torch.load(chkpt_path, map_location='cpu')
        load_model(model_g, checkpoint['model_g'], tag='model_g(resume)', logger=logger)
        load_model(model_d, checkpoint['model_d'], tag='model_d(resume)', logger=logger)

        optim_g.load_state_dict(checkpoint['optim_g'])
        optim_d.load_state_dict(checkpoint['optim_d'])

        saved_epoch = int(checkpoint['epoch'])
        step = int(checkpoint.get('step', 0))
        val_index = int(checkpoint.get('val_index', 0))
        epoch_wav_step_resume = int(checkpoint.get('epoch_wav_step', 0))
        resumed_mid_epoch = epoch_wav_step_resume > 0

        if resumed_mid_epoch:
            init_epoch = saved_epoch
            run_head_validate_epoch = saved_epoch
        else:
            init_epoch = saved_epoch + 1
            run_head_validate_epoch = saved_epoch

        run_head_validate = True
        run_head_validate_reason = "resume"

        if rank == 0:
            logger.info(
                f"[RESUME] saved_epoch={saved_epoch} init_epoch={init_epoch} step={step} "
                f"epoch_wav_step={epoch_wav_step_resume} val_index={val_index}"
            )
            logger.info(
                "[RESUME] optimizer lr_g=%.10f lr_d=%.10f" % (
                    optim_g.param_groups[0]['lr'], optim_d.param_groups[0]['lr']
                )
            )
            if 'epoch_wav_step' not in checkpoint:
                logger.warning(
                    "[RESUME] checkpoint has no epoch_wav_step; cannot resume in the middle of an epoch exactly. "
                    "Will restart from the head of the saved epoch."
                )
            if hp_str != checkpoint.get('hp_str', ''):
                logger.warning("New hparams is different from checkpoint. Will use new.")

    elif os.path.isfile(hp.train.pretrain):
        if rank == 0:
            logger.info("Start from 32k pretrain model: %s" % hp.train.pretrain)

        checkpoint = torch.load(hp.train.pretrain, map_location='cpu')
        load_model(model_g, checkpoint['model_g'], tag='model_g(pretrain)', logger=logger)
        load_model(model_d, checkpoint['model_d'], tag='model_d(pretrain)', logger=logger)

        init_epoch = 1
        step = 0
        val_index = 0

        run_head_validate = True
        run_head_validate_epoch = 0
        run_head_validate_reason = "pretrain"

    else:
        if rank == 0:
            logger.info("Starting new training run.")

        init_epoch = 1
        step = 0
        val_index = 0

        run_head_validate = True
        run_head_validate_epoch = 0
        run_head_validate_reason = "fresh"

    if args.num_gpus > 1:
        torch.distributed.barrier()

    if args.num_gpus > 1:
        model_g = DistributedDataParallel(model_g, device_ids=[rank])
        model_d = DistributedDataParallel(model_d, device_ids=[rank])

    torch.backends.cudnn.benchmark = True

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g,
        gamma=hp.train.lr_decay,
        last_epoch=init_epoch - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d,
        gamma=hp.train.lr_decay,
        last_epoch=init_epoch - 2
    )

    stft_criterion = MultiResolutionSTFTLoss(device, eval(hp.mrd.resolutions))
    spkc_criterion = nn.CosineEmbeddingLoss()

    trainloader = create_dataloader_train(hp, args.num_gpus, rank)

    if rank == 0 and run_head_validate:
        logger.info(
            f"Running {run_head_validate_reason} validation at step={step}, epoch={run_head_validate_epoch}"
        )
        with torch.no_grad():
            validate(
                hp, args, model_g, model_d, valloader, stft, writer,
                step, device, run_head_validate_epoch, val_index=val_index
            )

    if chkpt_path is not None and init_epoch > hp.train.epochs:
        if rank == 0:
            logger.info(
                f"[RESUME] checkpoint is already beyond target epochs: init_epoch={init_epoch}, target={hp.train.epochs}. Nothing to do."
            )
        return

    for epoch in range(init_epoch, hp.train.epochs + 1):

        if hasattr(trainloader, "batch_sampler") and hasattr(trainloader.batch_sampler, "set_epoch"):
            trainloader.batch_sampler.set_epoch(epoch)
        elif hasattr(trainloader, "sampler") and hasattr(trainloader.sampler, "set_epoch"):
            trainloader.sampler.set_epoch(epoch)

        loader = tqdm.tqdm(trainloader, desc='Loading train data') if rank == 0 else trainloader

        model_g.train()
        model_d.train()

        current_epoch_wav_step = 0
        skip_until = epoch_wav_step_resume if (resumed_mid_epoch and epoch == init_epoch) else 0

        if rank == 0:
            logger.info(
                "[EPOCH_START] epoch=%d step=%d skip_until=%d lr_g=%.10f lr_d=%.10f" % (
                    epoch, step, skip_until, optim_g.param_groups[0]['lr'], optim_d.param_groups[0]['lr']
                )
            )

        for ppg, ppg_l, vec, pit, spk, spec, spec_l, audio, audio_l, eng, deng, prd, flat in loader:
            if current_epoch_wav_step < skip_until:
                current_epoch_wav_step += 1
                continue

            ppg = ppg.to(device)
            vec = vec.to(device)
            pit = pit.to(device)
            spk = spk.to(device)
            spec = spec.to(device)
            audio = audio.to(device)
            ppg_l = ppg_l.to(device)
            spec_l = spec_l.to(device)
            audio_l = audio_l.to(device)
            eng = eng.to(device)
            deng = deng.to(device)
            prd = prd.to(device)
            flat = flat.to(device)

            if ppg.size(0) != 1:
                raise RuntimeError(
                    f"Train dataloader must yield exactly 1 wav per outer step. Got batch={ppg.size(0)}"
                )

            wav_segments = split_one_wav_to_segments(
                ppg, ppg_l, vec, pit, spk, spec, spec_l, audio, audio_l,
                eng, deng, prd, flat, hp
            )

            if not wav_segments:
                current_epoch_wav_step += 1
                continue

            for seg in wav_segments:
                fake_audio, sliced_audio, loss_g, gstats = run_one_segment(
                    seg, model_g, model_d, stft, stft_criterion, spkc_criterion, hp, device
                )
                loss_g.backward()

                if ((step + 1) % hp.train.accum_step == 0) or (step + 1 == len(loader)):
                    for param in model_g.parameters():
                        if param.grad is not None:
                            param.grad /= hp.train.accum_step
                    clip_grad_value_(model_g.parameters(), None)
                    optim_g.step()
                    optim_g.zero_grad()

                optim_d.zero_grad()
                loss_d = run_discriminator(fake_audio, sliced_audio, model_d)
                loss_d.backward()
                clip_grad_value_(model_d.parameters(), None)
                optim_d.step()

            step += 1
            current_epoch_wav_step += 1

            loss_g_item = gstats["loss_g_item"]
            loss_d_item = float(loss_d.item())
            loss_s_item = gstats["loss_s_item"]
            loss_m_item = gstats["loss_m_item"]
            loss_k_item = gstats["loss_k_item"]
            loss_r_item = gstats["loss_r_item"]
            loss_i_item = gstats["loss_i_item"]

            if rank == 0 and step % hp.log.info_interval == 0:
                writer.log_training(
                    loss_g_item, loss_d_item, loss_m_item, loss_s_item,
                    loss_k_item, loss_r_item, gstats["score_item"], step
                )
                logger.info(
                    "epoch %d | g %.04f m %.04f s %.04f d %.04f k %.04f r %.04f i %.04f | step %d | epoch_wav_step %d | wav_segments %d" % (
                        epoch, loss_g_item, loss_m_item, loss_s_item, loss_d_item,
                        loss_k_item, loss_r_item, loss_i_item, step, current_epoch_wav_step, len(wav_segments)
                    )
                )

            if rank == 0 and step % hp.log.save_interval == 0:
                step_path = os.path.join(pth_dir, f"{args.name}_{epoch:04d}_step{step:08d}.pt")
                val_index = save_and_validate(
                    rank, logger, model_g, model_d, optim_g, optim_d, args, hp_str,
                    step, epoch, step_path, hp, valloader, stft, writer, device,
                    val_index, current_epoch_wav_step
                )

        if rank == 0:
            epoch_path = resolve_resume_safe_save_path(pth_dir, args.name, epoch, chkpt_path)
            val_index = save_and_validate(
                rank, logger, model_g, model_d, optim_g, optim_d, args, hp_str,
                step, epoch, epoch_path, hp, valloader, stft, writer, device,
                val_index, 0
            )
            keep_ckpts = getattr(hp.log, 'keep_ckpts', 0)
            clean_checkpoints(pth_dir, args.name, keep_ckpts, logger)

        scheduler_g.step()
        scheduler_d.step()
