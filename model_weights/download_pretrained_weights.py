# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan
#
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser
from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download

MODEL_LIST = [
    # supervised models
    "sup.musdb18hq.bslocoformer-large",
    "sup.musdb18hq.bslocoformer-large.mixit-fma-large-pretrained",
    "sup.musdb18hq.bslocoformer-medium",
    "sup.musdb18hq.bslocoformer-medium.mixit-fma-full-pretrained",
    "sup.musdb18hq.bslocoformer-medium.mixit-fma-large-pretrained",
    "sup.musdb18hq.bslocoformer-medium.mixit-fma-medium-pretrained",
    "sup.musdb18hq.bslocoformer-medium.mixit-fma-small-pretrained",
    "sup.musdb18hq.bslocoformer-small",
    "sup.musdb18hq.bslocoformer-small.mixit-fma-large-pretrained",
    # mixit pretrained models
    "mixit.fma-full.bslocoformer-medium.12outputs",
    "mixit.fma-full.bslocoformer-medium.4outputs",
    "mixit.fma-large.bslocoformer-large.12outputs",
    "mixit.fma-large.bslocoformer-large.4outputs",
    "mixit.fma-large.bslocoformer-medium.12outputs",
    "mixit.fma-large.bslocoformer-medium.4outputs",
    "mixit.fma-large.bslocoformer-small.12outputs",
    "mixit.fma-large.bslocoformer-small.4outputs",
    "mixit.fma-medium.bslocoformer-medium.12outputs",
    "mixit.fma-medium.bslocoformer-medium.4outputs",
    "mixit.fma-small.bslocoformer-medium.12outputs",
    "mixit.fma-small.bslocoformer-medium.4outputs",
]

REPO_ID = "kohei0209/mixit_mss"
COMMIT_ID = "20a286f28b4a4c9058b935544cb7934ecc132443"


def download_model(dst_dir: Path, model_name: str):
    # model download
    downloaded_model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=f"models/{dst_dir.name}/{model_name}/checkpoints/model.pth",
        revision=COMMIT_ID,
    )
    dst_model_path = dst_dir / model_name / "checkpoints" / "model.pth"
    dst_model_path.parent.mkdir(exist_ok=True, parents=True)
    if dst_model_path.exists():
        print(f"{str(dst_model_path)} is skipped as it exists")
    else:
        shutil.copyfile(downloaded_model_path, dst_model_path)

    # config download
    downloaded_config_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=f"models/{dst_dir.name}/{model_name}/merged_config.yaml",
        revision=COMMIT_ID,
    )
    dst_config_path = dst_dir / model_name / "merged_config.yaml"
    if dst_config_path.exists():
        print(f"{str(dst_config_path)} is skipped as it exists")
    else:
        shutil.copyfile(downloaded_config_path, dst_config_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dst_dir", type=Path, default=Path(__file__).resolve().parent)
    args, unk_args = parser.parse_known_args()

    for model_name in MODEL_LIST:
        training_method = model_name.split(".")[0]  # sup or mixit
        dst_dir = args.dst_dir / training_method
        download_model(dst_dir=dst_dir, model_name=model_name)
