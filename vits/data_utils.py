import os
import numpy as np
import torch
import torch.utils.data

from vits.utils import load_wav_to_torch


def load_filepaths(filename, split="|"):
    with open(filename, encoding="utf-8") as f:
        filepaths = [line.strip().split(split) for line in f]
    return filepaths


class TextAudioSpeakerSet(torch.utils.data.Dataset):
    def __init__(self, filename, hparams):
        self.items = load_filepaths(filename)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.segment_size = hparams.segment_size
        self.hop_length = hparams.hop_length
        self._filter()
        print(f"----------{len(self.items)}----------")

    def _filter(self):
        lengths = []
        items_new = []
        for wavpath, spec, pitch, vec, ppg, spk, eng, deng, prd, flat in self.items:
            if not all(os.path.isfile(p) for p in [wavpath, spec, pitch, vec, ppg, spk, eng, deng, prd, flat]):
                continue
            temp = np.load(pitch)
            usel = int(temp.shape[0] - 1)
            if usel <= 0:
                continue
            items_new.append([wavpath, spec, pitch, vec, ppg, spk, eng, deng, prd, flat, usel])
            lengths.append(usel)
        pairs = list(zip(items_new, lengths))
        pairs.sort(key=lambda x: x[1])
        self.items = [x[0] for x in pairs]
        self.lengths = [x[1] for x in pairs]

    def read_wav(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        assert sampling_rate == self.sampling_rate, f"error: this sample rate of {filename} is {sampling_rate}"
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        return audio_norm

    def __getitem__(self, index):
        item = self.items[index]
        wav = self.read_wav(item[0])
        spe = torch.load(item[1])
        pit = np.load(item[2])
        vec = np.load(item[3])
        ppg = np.load(item[4])
        spk = np.load(item[5])
        eng = np.load(item[6])
        deng = np.load(item[7])
        prd = np.load(item[8])
        flat = np.load(item[9])

        vec = np.repeat(vec, 2, 0)
        ppg = np.repeat(ppg, 2, 0)

        pit = torch.FloatTensor(pit)
        vec = torch.FloatTensor(vec)
        ppg = torch.FloatTensor(ppg)
        spk = torch.FloatTensor(spk)
        eng = torch.FloatTensor(eng)
        deng = torch.FloatTensor(deng)
        prd = torch.FloatTensor(prd)
        flat = torch.FloatTensor(flat)

        len_min = min(
            pit.size(0), vec.size(0) - 2, ppg.size(0) - 2, spe.size(1),
            wav.size(1) // self.hop_length, eng.size(0), deng.size(0), prd.size(0), flat.size(0)
        )
        wav_len = len_min * self.hop_length

        pit = pit[:len_min]
        vec = vec[:len_min, :]
        ppg = ppg[:len_min, :]
        spe = spe[:, :len_min]
        wav = wav[:, :wav_len]
        eng = eng[:len_min]
        deng = deng[:len_min]
        prd = prd[:len_min]
        flat = flat[:len_min]
        return spe, wav, ppg, vec, pit, spk, eng, deng, prd, flat

    def __len__(self):
        return len(self.items)


class TextAudioSpeakerCollate:
    def __call__(self, batch):
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].size(1) for x in batch]), dim=0, descending=True
        )

        max_spe_len = max([x[0].size(1) for x in batch])
        max_wav_len = max([x[1].size(1) for x in batch])
        spe_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        spe_padded = torch.FloatTensor(len(batch), batch[0][0].size(0), max_spe_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        spe_padded.zero_(); wav_padded.zero_()

        max_ppg_len = max([x[2].size(0) for x in batch])
        ppg_lengths = torch.LongTensor(len(batch))
        ppg_padded = torch.FloatTensor(len(batch), max_ppg_len, batch[0][2].size(1))
        vec_padded = torch.FloatTensor(len(batch), max_ppg_len, batch[0][3].size(1))
        pit_padded = torch.FloatTensor(len(batch), max_ppg_len)
        eng_padded = torch.FloatTensor(len(batch), max_ppg_len)
        deng_padded = torch.FloatTensor(len(batch), max_ppg_len)
        prd_padded = torch.FloatTensor(len(batch), max_ppg_len)
        flat_padded = torch.FloatTensor(len(batch), max_ppg_len)
        ppg_padded.zero_(); vec_padded.zero_(); pit_padded.zero_(); eng_padded.zero_(); deng_padded.zero_(); prd_padded.zero_(); flat_padded.zero_()
        spk = torch.FloatTensor(len(batch), batch[0][5].size(0))

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            spe_padded[i, :, :row[0].size(1)] = row[0]
            spe_lengths[i] = row[0].size(1)
            wav_padded[i, :, :row[1].size(1)] = row[1]
            wav_lengths[i] = row[1].size(1)
            ppg_padded[i, :row[2].size(0), :] = row[2]
            ppg_lengths[i] = row[2].size(0)
            vec_padded[i, :row[3].size(0), :] = row[3]
            pit_padded[i, :row[4].size(0)] = row[4]
            spk[i] = row[5]
            eng_padded[i, :row[6].size(0)] = row[6]
            deng_padded[i, :row[7].size(0)] = row[7]
            prd_padded[i, :row[8].size(0)] = row[8]
            flat_padded[i, :row[9].size(0)] = row[9]

        return (
            ppg_padded, ppg_lengths, vec_padded, pit_padded, spk,
            spe_padded, spe_lengths, wav_padded, wav_lengths,
            eng_padded, deng_padded, prd_padded, flat_padded,
        )
