import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vits.data_utils import TextAudioSpeakerSet
from vits_extend.dataloader import create_dataloader_train
from omegaconf import OmegaConf


def main():
    hp = OmegaConf.load("configs/base.yaml")

    dataset = TextAudioSpeakerSet(hp.data.training_files, hp.data)
    print("dataset size:", len(dataset))

    loader = create_dataloader_train(hp, n_gpus=1, rank=0)

    for i, batch in enumerate(loader):
        print(f"batch {i} OK")
        if i > 5:
            break


if __name__ == "__main__":
    main()