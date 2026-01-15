<!--
Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan

SPDX-License-Identifier: MIT
-->

# Unsupervised Pre-training of MSS models Using MixIT

[![IEEE WASPAA DOI](https://img.shields.io/badge/IEEE–WASPAA-10.1109/WASPAA66052.2025.11231018-blue.svg)](https://doi.org/10.1109/WASPAA66052.2025.11231018)
[![arXiv](https://img.shields.io/badge/arXiv-2505.07631-b31b1b.svg)](https://arxiv.org/abs/2505.07631)


This repository includes source code for the following WASPAA 2025 paper.
If you use any part of this code including the pre-trained models for your work, please cite the paper:
```
@InProceedings{Saijo2025_MixITMSS,
  author    =  {Saijo, Kohei and Bando, Yoshiaki},
  title     =  {Is MixIT Really Unsuitable for Correlated Sources? Exploring MixIT for Unsupervised Pre-training in Music Source Separation},
  booktitle =  {Proc. Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)},
  year      =  2025,
  month     =  oct
}
```

## 1. Installation

```sh
# install this repo
git clone https://github.com/b-sigpro/mixit-mss.git
cd mixit-mss
pip install -e .
pip install -r requirements.txt

# install aiaccel
git clone https://github.com/aistairc/aiaccel.git
cd aiaccel
git checkout 117d8d5d335540b6d331993ffc02d4b64f5e02a1  # the commit where we tested our code
pip install -e .
cd ../
```

## 2. Pre-trained models

We provide the pre-trained models on a [Hugging Face repository](https://huggingface.co/kohei0209/mixit_mss), which can be downloaded by running a python script:
```sh
python model_weights/download_pretrained_weights.py
```
By default, it makes the `model_weights` directory and download models under it.
The detailed description is provided below.



### MixIT pre-trained models
 `./model_weights/mixit` directory contains MixIT pre-trained models (without supervised fine-tuning).

Each directory is named as `mixit.fma-${dataset_size}.bslocoformer-${model_size}.${num_outputs}outputs`, where:
- `dataset_size`: The size of the FMA dataset used for pre-training (`small`, `medium`, `large`, or `full`).
- `model_size`: The size of the separation model, BS-Locoformer (`small`, `medium`, or `large`).
- `num_outputs`: The number of the separator's outputs (`12` or `4`). The original pre-trained models have 12 output channels, but we selected 4 out of 12 before supervised fine-tuning (see the paper for more details). 

```sh
./model_weights/mixit
├── mixit.fma-full.bslocoformer-medium.12outputs
│   ├── checkpoints
│   │   └── model.pth
│   └── merged_config.yaml
├── mixit.fma-full.bslocoformer-medium.4outputs
│   ├── checkpoints
│   │   └── model.pth
│   └── merged_config.yaml
...
```

### Fine-tuned models

 `./model_weights/sup` directory contains the models fine-tuned on the MUSDB18HQ dataset.

 Each directory is named as `sup.musdb18hq.bslocoformer-${model_size}.mixit-fma-${dataset_size}-pretrained`.
 The directories without `.mixit-fma-${dataset_size}-pretrained` suffix contain model weights trained from scratch without pre-training.

 ```sh
./model_weights/sup
├── sup.musdb18hq.bslocoformer-large
│   ├── checkpoints
│   │   └── model.pth
│   └── merged_config.yaml
├── sup.musdb18hq.bslocoformer-large.mixit-fma-large-pretrained
│   ├── checkpoints
│   │   └── model.pth
│   └── merged_config.yaml
...
```

### Separating a music mixture with a pre-trained model

The fine-tuned model can be used to separate a musical mixture into bass, drums, vocals, and other instruments.

The command below is an example to separate a audio file with the medium BS-Locoformer model, with 12-second chunk size and 6-second overlap:
```sh
python separate_sample.py model_weights/sup/sup.musdb18hq.bslocoformer-medium.mixit-fma-large-pretrained /path/to/audio-file /path/to/output-directory --css_segment_size 12 --css_shift_size 6 
```




## 3. Pre-training using MixIT on the FMA dataset
Assume you are now at `./mixit-mss`.

### Data preparation

We provide a shell script, `data.sh`, to easily prepare the data. `data.sh` does the following processes:

1. Download the FMA metadata and unzip it
2. Download the FMA data with the specified size and unzip it
3. Segment the FMA audio files and make an HDF5 file under `recipes/fma/hdf5`

It can be run as:
```sh
fma_size=large  # FMA data size (small, medium, large, or full)
np=1  # The number of parallel processes when making the HDF5. np>1 leads to faster data preparation but may cause the process to be stuck.
./recipes/fma/scripts/data.sh ${fma_size} ${np}
```

### Training
The main script for training is in `train.py`, which can be run by
```sh
python -m aiaccel.torch.apps.train /path/to/config.yaml
```
We provide several configuration files  `recipes/fma/models/mixit.fma-${data_size}.bslocoformer-${model_size}/config.yaml`.

The main configuration file is `recipes/fma/models/mixit.fma-large.bslocoformer-medium/config.yaml`, and the other files inherits this file while overwriting only some specific parts.

After training, the directory structure should look like as follows. The lightning's checkpoints are saved under `checkpoints`. The training progress can be watched with Tensorboard.
```
recipes/fma/models/mixit.fma-small.bslocoformer-medium
├── checkpoints
├── config.yaml
├── events.out.tfevents.xxx.xxx.xxx.x
├── hparams.yaml
├── log.txt
├── merged_config.yaml
└── train.sh
```


## 4. Supervised fine-tuning on the MUSDB18HQ dataset

### Data preparation

We provide a shell script, `data.sh`, to easily prepare the data. `data.sh` does the following processes:

1. Download MUSDB18HQ unzip it
2. Split training and validation set following the common split
3. Apply unsupervised source activity detection (introduced in the BSRNN paper) to the training data and save the segmented audio files as an HDF5 file
4. Segment the validation data and save as an HDF5 file (which will be used when performing the output-channel selection)

It can be run as:
```sh
./recipes/musdb18hq/scripts/data.sh
```

### Output-channel selection of MixIT-pretrained model

Since MixIT-pretrained model typically has more output channels than we actually want, we need to select some output channels. For instance, our model has 12 output channels while only 4 of them are necessary in the VDBO setup.

We provide a script that selects 4 output channels using the validation set of the MUSDB18HQ:
```sh
./recipes/musdb18hq/scripts/select_heads.sh /path/to/.ckpt
```


### Training

Training on MUSDB18HQ can be in a similar manner to that on FMA:
```sh
python -m aiaccel.torch.apps.train /path/to/config.yaml
```

The directory strucure after training is similar to that of FMA.

### Evaluation

Once you finish the training, running `separate.sh` runs inference and scoring:
```sh
./recipes/musdb18hq/scripts/separate.sh /path/to/model_directory
```
Here, `/path/to/model_directory` can be either a path to directory including the `checkpoints` directory or a direct path to `.ckpt` file.
For instance, when evaluating `recipes/musdb18hq/models/sup.musdb18hq.bslocoformer-medium`, you can give `recipes/musdb18hq/models/sup.musdb18hq.bslocoformer-medium` or `recipes/musdb18hq/models/sup.musdb18hq.bslocoformer-medium/checkpoints/xxx.ckpt`.
In the former case, all the checkpoints under that directory except for `last.ckpt` are averaged, and the averaged parameters are used for evaluation.

The segment and shift size in inference are by default set to 12 and 6 seconds, respectively.
One can change these configurations by giving them as the second and third arguments when running `separate.sh`.



## 5. Copyright and license

Released under `MIT` license, as found in the [LICENSE.md](LICENSE.md) file.

All files, except as noted below:

```
Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan

SPDX-License-Identifier: MIT
```

The following file:

- `mixit_mss/mixit/model/bslocoformer.py`

was adapted from https://github.com/merlresearch/tf-locoformer (license included in [LICENSES/Apache-2.0.md](LICENSES/Apache-2.0.md))

```
Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan
Copyright (c) 2024 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: MIT
SPDX-License-Identifier: Apache-2.0
```

The following file:

- `mixit_mss/mixit/model/average_model_params.py`

was adapted from https://github.com/espnet/espnet (license included in [LICENSES/Apache-2.0.md](LICENSES/Apache-2.0.md))

```
Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan
Copyright (c) 2017 ESPnet Developers

SPDX-License-Identifier: MIT
SPDX-License-Identifier: Apache-2.0
```

The following files:

- `mixit_mss/mixit/loss/efficient_mixit.py`
- `mixit_mss/mixit/loss/snr.py`

were adapted from https://github.com/kohei0209/self-remixing (license included in [LICENSES/MIT.md](LICENSES/MIT.md))

```
Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan
Copyright (c) 2024 Kohei Saijo

SPDX-License-Identifier: MIT
SPDX-License-Identifier: MIT
```

The following file:

- `mixit_mss/mixit/loss/pit_wrapper.py`

was adapted from https://github.com/asteroid-team/asteroid (license included in [LICENSES/MIT.md](LICENSES/MIT.md))

```
Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan
Copyright (c) 2019 Pariente Manuel

SPDX-License-Identifier: MIT
SPDX-License-Identifier: MIT
```