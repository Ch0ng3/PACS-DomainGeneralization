# Which pretraining generalises? Vision backbones under domain shift

Deep Learning, Project 5 (DomainBed on PACS), Milestone 1.

## Team

Team name: PACS Team

| Name | Neptun code |
|---|---|
| Márton Tamás Magyar | `TG2UTM` |
| Daniel Brody | `IASULA` |
| Marco Buonfantino | `FMDKOH` |

## Project description

A classifier trained on photos should also recognise the same objects in paintings, cartoons and sketches, but most models cannot. In this project we compare backbones pretrained in different ways (supervised ResNet-50, self-supervised DINOv2 and language-supervised CLIP) with DomainBed on PACS. PACS has 7 object classes (dog, elephant, giraffe, guitar, horse, house, person) in 4 styles, called domains: art painting, cartoon, photo and sketch. We evaluate leave-one-domain-out: the model is trained on 3 domains and tested on the 4th, which it never sees during training or model selection.

Milestone 1 is about the data: where it comes from and how we downloaded it, what it looks like, how we prepared it, and the training, validation and test inputs and outputs. The repository also has our reproduction of the DomainBed ERM baseline, which uses the same data.

## How the files fit together

`Milestone1_ERM_reproduction.ipynb` is the full pipeline we ran on Colab: it downloads PACS to `/content/data/PACS`, lets DomainBed split it and trains ERM, which gives the results below. DomainBed does the data steps inside its own code, so they do not show up in that notebook.

`prepare_data.py` and `Milestone1_data_preparation.ipynb` repeat those data steps outside DomainBed, so that each one can be seen and checked: the download, the exploration and plots, and the train/validation/test splits saved as CSV files. The splits follow DomainBed's split code with the same seeds, so they match the data the ERM runs used. These two files do not train any model, and the ERM results do not depend on them.

## Data source and download

- Dataset: PACS, 9,991 images (227 × 227 RGB) of 7 classes in 4 domains: 2,048 art paintings, 2,344 cartoons, 1,670 photos and 3,929 sketches. Paper: https://arxiv.org/abs/1710.03077
- Source: the Hugging Face copy given in the project description, https://huggingface.co/datasets/flwrlabs/pacs. DomainBed's own download (`download_pacs`, from Google Drive with `gdown`) failed in our sessions because of the Google Drive quota.
- Download: `prepare_data.py` loads the dataset with `load_dataset("flwrlabs/pacs", split="train")` from the Hugging Face `datasets` library and saves each image as an RGB JPEG in DomainBed's folder layout, `data/PACS/<domain>/<class>/<00000>.jpg`. The images are not stored in this repository.

## What the exploration shows

- The domains have different sizes, from 1,670 images (photo) to 3,929 (sketch).
- The class distribution changes from domain to domain. In sketch there are only 80 houses and 160 persons, while person is the largest class in art painting and photo and one of the largest in cartoon.
- The pixel statistics change too. Sketches are black lines on white (mean gray level 0.96, no color), cartoons are bright with flat colors, and art paintings are the most colorful. For this reason we normalize with the ImageNet mean and std, which the pretrained backbones expect, and not with the statistics of one domain.
- No two files are identical (checked with MD5 hashes).

## Data preparation and final output

For every held-out (test) domain there is a folder `data/splits/test_<domain>/` with three files:

| file | content | used for |
|---|---|---|
| `train.csv` | `in` part (80%) of the 3 training domains | training |
| `val.csv` | `out` part (20%) of the 3 training domains | model selection (training-domain validation) |
| `test.csv` | all images of the held-out domain | final evaluation only |

The 80/20 split follows DomainBed's code (`--holdout_fraction 0.2`, seed `seed_hash(trial_seed=0, env_i)`), so it matches the split DomainBed uses in training. Each row has the columns `path, domain, class_name, label, part`.

The CSV files hold only paths and labels, not images. On Colab, `download_pacs` first writes the images to `data/PACS/` inside the cloned repository, and the paths in the CSV files (for example `PACS/sketch/dog/00000.jpg`) are relative to `data/`. `data/pacs_index_parts.csv` has all 9,991 images with their part; the split files are subsets of it.

In `test.csv` the `part` column still matters: DomainBed also splits the held-out domain 80/20 and reports the accuracy on its `in` part. So the ERM results below are measured on the rows of `test.csv` with `part = in` (we checked this against the training logs in the ERM notebook).

| test domain | train | val | test |
|---|---|---|---|
| art_painting | 6,356 | 1,587 | 2,048 |
| cartoon | 6,119 | 1,528 | 2,344 |
| photo | 6,659 | 1,662 | 1,670 |
| sketch | 4,851 | 1,211 | 3,929 |

`load_split(test_domain, split)` in `prepare_data.py` turns a split into model inputs and outputs: it reads the CSV, opens each image from `data/PACS/` and preprocesses it. The inputs `X` are float32 arrays of shape `(N, 3, 224, 224)`, resized to 224 and normalized with the ImageNet statistics, with DomainBed-style augmentation for the training split. The outputs `y` are the int64 class labels 0 to 6, shape `(N,)`.

## ERM baseline on these data

We trained ERM with an ImageNet ResNet-50 (`resnet50_augmix` switched off, as in the paper), default hyperparameters and seed 0. The table gives the accuracy on the held-out domain, with the model chosen on the training-domain validation set:

| | Art | Cartoon | Photo | Sketch | Average |
|---|---|---|---|---|---|
| Ours | 82.5 | 80.9 | 97.8 | 78.3 | 84.9 |
| Paper (DomainBed) | 84.7 | 80.8 | 97.2 | 79.3 | 85.5 |

The paper reports the best result of a random hyperparameter search, averaged over 3 trials. With one run and the default hyperparameters, a gap of 1 to 2 points per domain is expected.

## Files in the repository

| file / folder | what it does |
|---|---|
| `prepare_data.py` | Python script. Downloads PACS, builds the image index, computes the exploration tables and figures, writes the train/val/test splits and checks them. `load_split` returns the inputs `X` and outputs `y` of a split. |
| `Milestone1_data_preparation.ipynb` | Jupyter Notebook with the same steps as `prepare_data.py`, one at a time, with plots and comments. It has already been run from a fresh clone, with the images downloaded from Hugging Face, and the outputs are saved. No GPU is needed. |
| `Milestone1_ERM_reproduction.ipynb` | Jupyter Notebook we ran on a Google Colab GPU. It clones DomainBed, switches off `resnet50_augmix`, downloads PACS, trains ERM 4 times (once per held-out domain) and collects the results. |
| `data/class_names.json` | class names in label order (0 is dog, 6 is person) |
| `data/pacs_index.csv` | one row per image: path, domain, class, label, size, brightness, contrast, colorfulness, MD5 |
| `data/pacs_index_parts.csv` | one row per image with its DomainBed part (`in` 80%, `out` 20%) |
| `data/splits/test_<domain>/{train,val,test}.csv` | the final output: the 4 leave-one-domain-out settings |
| `data/stats/` | class counts, pixel statistics per domain, duplicate report |
| `figures/` | `class_distribution.png`, `samples.png` (one example per domain and class), `pixel_stats.png` |

## How to run

Everything runs on Google Colab. The outputs (split files, statistics, figures, executed notebooks) are already in the repository, and running the code again gives the same split files.

Jupyter Notebook: open `Milestone1_data_preparation.ipynb` in Colab and run all cells. The first cell clones this repository and installs `datasets`; a CPU runtime is enough.

Python script: in a Colab cell, run

```
!git clone https://github.com/Ch0ng3/PACS-DomainGeneralization.git
%cd PACS-DomainGeneralization
!pip -q install datasets
!python prepare_data.py    # data in ./data, figures in ./figures
```

ERM reproduction: open `Milestone1_ERM_reproduction.ipynb` on Google Colab with a GPU and run all cells. Each run takes about 42 minutes, about 2.8 hours for all four.

## Related materials

From the project description:

- Code: https://github.com/facebookresearch/DomainBed
- Code: https://github.com/facebookresearch/dinov2 and https://github.com/mlfoundations/open_clip
- Data, PACS (fallback copy): https://huggingface.co/datasets/flwrlabs/pacs
- Paper, DomainBed: https://arxiv.org/abs/2007.01434
- Paper, PACS: https://arxiv.org/abs/1710.03077
- Paper, DINOv2: https://arxiv.org/abs/2304.07193
- Paper, CLIP: https://arxiv.org/abs/2103.00020
- Paper, WiSE-FT: https://arxiv.org/abs/2109.01903
