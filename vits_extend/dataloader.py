from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from vits.data_utils import TextAudioSpeakerCollate
from vits.data_utils import TextAudioSpeakerSet


def create_dataloader_train(hps, n_gpus, rank):
    collate_fn = TextAudioSpeakerCollate()
    train_dataset = TextAudioSpeakerSet(hps.data.training_files, hps.data)

    # outer step = 1 wav
    wav_batch_size = 1

    if n_gpus > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=n_gpus,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        train_loader = DataLoader(
            train_dataset,
            num_workers=4,
            shuffle=False,
            sampler=train_sampler,
            batch_size=wav_batch_size,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            num_workers=4,
            shuffle=False,
            batch_size=wav_batch_size,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

    return train_loader


def create_dataloader_eval(hps):
    collate_fn = TextAudioSpeakerCollate()
    eval_dataset = TextAudioSpeakerSet(hps.data.validation_files, hps.data)
    eval_loader = DataLoader(
        eval_dataset,
        num_workers=2,
        shuffle=False,
        batch_size=1,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    return eval_loader
