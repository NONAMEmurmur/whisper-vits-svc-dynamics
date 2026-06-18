import tqdm
import torch
import torch.nn.functional as F


def validate(hp, args, generator, discriminator, valloader, stft, writer, step, device, epoch, val_index=0):
    generator.eval()
    discriminator.eval()
    torch.backends.cudnn.benchmark = False

    num_batches = len(valloader)
    if num_batches == 0:
        writer.log_validation(0.0, generator, discriminator, step)
        torch.backends.cudnn.benchmark = True
        return

    target_batch_idx = val_index % num_batches
    mel_loss = 0.0
    found = False

    loader = tqdm.tqdm(valloader, desc=f"Validation loop (target batch={target_batch_idx})")

    with torch.no_grad():
        for idx, (ppg, ppg_l, vec, pit, spk, spec, spec_l, audio, audio_l, eng, deng, prd, flat) in enumerate(loader):
            if idx != target_batch_idx:
                continue

            found = True

            ppg = ppg.to(device)
            vec = vec.to(device)
            pit = pit.to(device)
            spk = spk.to(device)
            ppg_l = ppg_l.to(device)
            audio = audio.to(device)
            eng = eng.to(device)
            deng = deng.to(device)
            prd = prd.to(device)
            flat = flat.to(device)

            if hasattr(generator, "module"):
                fake_audio = generator.module.infer(
                    ppg, vec, pit, spk, ppg_l, eng, deng, prd, flat
                )[:, :, :audio.size(2)]
            else:
                fake_audio = generator.infer(
                    ppg, vec, pit, spk, ppg_l, eng, deng, prd, flat
                )[:, :, :audio.size(2)]

            mel_fake = stft.mel_spectrogram(fake_audio.squeeze(1))
            mel_real = stft.mel_spectrogram(audio.squeeze(1))
            mel_loss = F.l1_loss(mel_fake, mel_real).item()

            spec_fake = stft.linear_spectrogram(fake_audio.squeeze(1))
            spec_real = stft.linear_spectrogram(audio.squeeze(1))

            audio_np = audio[0][0].cpu().detach().numpy()
            fake_audio_np = fake_audio[0][0].cpu().detach().numpy()
            spec_fake_np = spec_fake[0].cpu().detach().numpy()
            spec_real_np = spec_real[0].cpu().detach().numpy()

            writer.log_fig_audio(audio_np, fake_audio_np, spec_fake_np, spec_real_np, 0, step)
            break

    if not found:
        print(
            f"[WARN] validation target batch not found: "
            f"val_index={val_index}, target_batch_idx={target_batch_idx}, num_batches={num_batches}"
        )

    writer.log_validation(mel_loss, generator, discriminator, step)
    torch.backends.cudnn.benchmark = True
