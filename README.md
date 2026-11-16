# Which pretraining generalises? Vision backbones under domain shift

Deep Learning, Project 5 (DomainBed on PACS).

## Team

Team name: PACS Team

| Name | Neptun code |
|---|---|
| Márton Tamás Magyar | TG2UTM |
| Daniel Brody | IASULA |
| Marco Buonfantino | FMDKOH |

## Project

A classifier trained on photos should also recognise the same objects in paintings, cartoons and sketches, but most models cannot. We compare backbones pretrained in different ways (supervised ResNet-50, self-supervised DINOv2 and language-supervised CLIP) with DomainBed on PACS: 7 object classes in 4 styles (art painting, cartoon, photo, sketch). Each model is trained on 3 styles and tested on the 4th, which it never sees during training or model selection.

## Repository structure

| Folder | Content |
|---|---|
| `Milestone_1/` | Milestone 1 - Reproduce: PACS data (source, exploration, train/val/test splits) and the DomainBed ERM ResNet-50 baseline |
| `Milestone_2/` | Milestone 2 - Improve: DINOv2 and CLIP backbones, adapted with linear probe, LoRA and full fine-tuning |

Each folder has its own README with the details, the instructions to run the code and the results. The notebooks are saved with their outputs. Everything runs on Google Colab.

## Milestone 1 - Reproduce

ERM with an ImageNet ResNet-50 (resnet50_augmix off), model chosen on the training-domain validation set: 84.9% average accuracy on the held-out domain, against 85.5% in the DomainBed paper. Details in the README of `Milestone_1/`.

## Milestone 2 - Improve

We replaced the ResNet-50 with DINOv2 and CLIP and compared three ways of adapting them, with the same split, protocol and metrics as Milestone 1. Average accuracy on the held-out domain (%):

| Backbone | Linear probe | LoRA | Full fine-tuning |
|---|---|---|---|
| ResNet-50 (baseline) | - | - | 84.9 |
| DINOv2 ViT-B/14 | 94.5 | 94.5 | 93.5 |
| CLIP ViT-B/16 | 96.7 | 97.1 | 95.4 |

CLIP zero-shot, without any training, reaches 97.2. What we changed and why, how we tested it, the full results table and the discussion are in the README of `Milestone_2/`.

## Related materials

- Code: https://github.com/facebookresearch/DomainBed
- Code: https://github.com/facebookresearch/dinov2 and https://github.com/mlfoundations/open_clip
- Data, PACS (fallback copy): https://huggingface.co/datasets/flwrlabs/pacs
- Paper, DomainBed: https://arxiv.org/abs/2007.01434
- Paper, PACS: https://arxiv.org/abs/1710.03077
- Paper, DINOv2: https://arxiv.org/abs/2304.07193
- Paper, CLIP: https://arxiv.org/abs/2103.00020
