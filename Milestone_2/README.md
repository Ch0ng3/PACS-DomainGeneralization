# Milestone 2 - Improve

Task from the project description:

- Replace the backbone with DINOv2 and CLIP models.
- Compare three ways of adapting them: a linear classifier on frozen features, LoRA, and full fine-tuning.

The baseline is the ERM ResNet-50 of Milestone 1, with 84.9% average accuracy on the held-out domain. We replaced its backbone with DINOv2 ViT-B/14 and CLIP ViT-B/16 and adapted each of them in three ways: linear probe, LoRA and full fine-tuning. This gives 6 configurations, each trained 4 times (one run per held-out domain), so 24 runs in total. We also evaluated CLIP zero-shot, which needs no training, as a reference.

All notebooks in this folder are saved with their outputs, and the numbers of every run are in `results/`, so the results can be checked without running the code.

## 1. What we changed and why

We made two changes to the baseline and tested them one at a time.

### Change 1: the backbone

| Backbone | Pretraining | Feature given to the classifier |
|---|---|---|
| ResNet-50 (baseline) | supervised, ImageNet-1k (about 1.3M photos) | 2048-d pooled feature |
| DINOv2 ViT-B/14 | self-supervised, LVD-142M (about 142M images) | class tokens of the last 4 blocks and the mean patch token, 3840-d in total |
| CLIP ViT-B/16 (OpenCLIP, LAION-2B) | trained on about 2B image-text pairs from the web | 512-d image embedding |

ImageNet is made almost only of photographs, so the ResNet-50 learns features that depend on photographic texture and color. In Milestone 1 it lost the most on sketch (78.3) and cartoon (80.9). DINOv2 and CLIP were pretrained on much more and much more varied data. CLIP in particular has seen drawings, paintings, cartoons and sketches together with a text that names the object, so it should recognise an object independently of how it is drawn. We therefore expected large gains on art, cartoon and sketch, a small gain on photo (already 97.8), and CLIP to be at least as good as DINOv2 on the non-photo domains.

### Change 2: how the backbone is adapted

| Method | What is trained |
|---|---|
| Linear probe | only a linear classifier, the backbone is frozen |
| LoRA | small rank-8 adapters on the attention layers (about 0.44M parameters) and the classifier, the backbone is frozen |
| Full fine-tuning | all the weights |

With the linear probe the pretrained features cannot change, so training on three styles cannot damage them, but they also cannot adapt to the 7 PACS classes. Full fine-tuning can adapt everything, but it may overwrite the general features with features tuned on the three training styles, which would hurt the unseen one. We expected LoRA to sit in between and to be the best compromise.

### Reference: CLIP zero-shot

CLIP can classify without training, by comparing the image with the text "a photo of a {class}." for each class. It has no hyperparameters and no model selection, so it shows how much the training on PACS really adds.

## 2. How we tested each change

Each comparison changes only one thing:

| Comparison | Kept fixed | Changed |
|---|---|---|
| Backbone | adaptation method | ResNet-50, DINOv2, CLIP |
| Adaptation | backbone | linear probe, LoRA, full fine-tuning |

What is the same for the baseline and every experiment:

| Setting | Value |
|---|---|
| Code | DomainBed with the ERM algorithm (each notebook adds the patches needed for CLIP and LoRA) |
| Data | PACS from Hugging Face, leave-one-domain-out: 4 runs per configuration, one per held-out domain |
| Split | the same split as Milestone 1 (holdout fraction 0.2, seed 0) |
| Training | DomainBed defaults: 32 images per training domain per step, DomainBed augmentation, 224 x 224 images, Adam |
| Model selection | a checkpoint every 300 steps; we keep the one with the best accuracy on the validation part (20%) of the 3 training domains |
| Test set | the held-out domain, used only for the final evaluation, never to choose a model or a hyperparameter |
| Hyperparameters | standard values for each method, fixed before the runs, no search |

Metrics, all on the held-out domain:

- Accuracy (%), the main metric, for each held-out domain: Art painting (A), Cartoon (C), Photo (P), Sketch (S).
- Avg: the mean of the 4 accuracies.
- ECE (%): calibration error, how far the confidence of the model is from its real accuracy (15 bins). Lower is better.
- Cost: number of trained parameters and GPU memory.

Training budget. Each method needs its own learning rate, because they train very different numbers of parameters:

| Method | Learning rate | Steps |
|---|---|---|
| ERM ResNet-50 (baseline) | 5e-5 | 5000 |
| Linear probe | 5e-5 | 2000 |
| LoRA | 5e-4 | 1500 |
| Full fine-tuning | 1e-5 (classifier 1e-4) | 1500 |

All new runs use fewer steps than the baseline and take a few minutes each on an A100 GPU, so the improvements do not come from a bigger training budget.

## 3. Results

Accuracy on the held-out domain (%), seed 0, model chosen on the training-domain validation set.

| # | Backbone | Method | What changed compared to the baseline | A | C | P | S | Avg | Diff. vs baseline | ECE (%) | Trained params | GPU memory |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B | ResNet-50 | ERM, all weights trained | baseline (Milestone 1) | 82.5 | 80.9 | 97.8 | 78.3 | 84.9 | - | - | 24M | about 8 GB |
| 1 | DINOv2 | linear probe | new backbone, frozen, only a linear classifier | 97.8 | 92.4 | 99.9 | 87.7 | 94.5 | +9.6 | 2.0 | 27k | about 1 GB |
| 2 | DINOv2 | LoRA | new backbone, frozen, with LoRA adapters | 96.2 | 95.6 | 100.0 | 86.4 | 94.5 | +9.6 | 3.4 | 0.47M | about 11 GB |
| 3 | DINOv2 | full fine-tuning | new backbone, all weights trained | 96.6 | 93.0 | 99.9 | 84.8 | 93.5 | +8.6 | 3.4 | 87M | about 17 GB |
| 4 | CLIP | zero-shot | new backbone, no training | 98.8 | 99.1 | 99.8 | 91.0 | 97.2 | +12.3 | - | 0 | - |
| 5 | CLIP | linear probe | new backbone, frozen, only a linear classifier | 96.8 | 98.5 | 99.4 | 92.1 | 96.7 | +11.8 | 21.0 | 4k | about 1 GB |
| 6 | CLIP | LoRA | new backbone, frozen, with LoRA adapters | 98.5 | 97.9 | 99.7 | 92.4 | 97.1 | +12.2 | 1.8 | 0.45M | about 9 GB |
| 7 | CLIP | full fine-tuning | new backbone, all weights trained | 96.6 | 96.3 | 99.0 | 89.6 | 95.4 | +10.5 | 2.5 | 112M | about 13 GB |

The ECE was not measured for the baseline and for CLIP zero-shot, and GPU memory was not measured for CLIP zero-shot, which only runs inference. There is no DINOv2 zero-shot row: DINOv2 has no text encoder, so it cannot classify images without training at least a classifier on top (row 1).

Every configuration is better than the baseline, but several changes did not help compared to the simpler option on the same backbone:

- LoRA on DINOv2 (2) is not better than the linear probe (1).
- Full fine-tuning (3 and 7) is worse than the linear probe and LoRA on both backbones.
- No trained CLIP model is clearly better than CLIP zero-shot (4).

We used one seed, so differences of less than 1-2 points should not be over-interpreted. The accuracies agree with DomainBed's own collect_results script (section 8 of each run notebook).

Before these final runs we tried two earlier versions. A first linear-probe notebook gave 95.1 (DINOv2) and 96.5 (CLIP), within 1 point of the final runs with the same settings. A first LoRA notebook was discarded because it put the adapters in different layers for DINOv2 and CLIP, so the two backbones were not comparable.

Where the numbers come from:

- `results/m2_runs.jsonl`: one line per run (24 lines), taken from the RESULT lines printed by the run notebooks. For each run: held-out domain, accuracy, ECE, mean confidence, selected step, trained parameters, GPU memory and confusion matrix (rows are the true class, columns the predicted class, in the order dog, elephant, giraffe, guitar, horse, house, person).
- `results/m2_summary.csv`: the table above as a CSV file.
- `M2_results.ipynb`: the final table, the confusion matrices, the calibration analysis and CLIP zero-shot.

## 4. What we learned

What helped:

- Changing the backbone is by far the biggest improvement, between +8.6 and +12.3 points. As expected, the gain is largest on the domains that are far from photos (up to about +16 on art, +18 on cartoon, +14 on sketch), while photo was already close to 100%.
- CLIP is better than DINOv2 with every method, mainly on cartoon and sketch, even if its feature is much smaller. This agrees with the idea that training with text teaches the model what the object is, independently of the style.
- LoRA on CLIP gives the best trained model (97.1) and also fixes the calibration problem of the CLIP linear probe (ECE from 21% to 1.8%), with less GPU memory than full fine-tuning.
- The linear probe is a very good and cheap option: 94.5 with DINOv2 and 96.7 with CLIP, with a few thousand trained parameters and very little GPU memory.

What did not help, and possible reasons:

- Full fine-tuning is the worst method on both backbones, about 1 point below the linear probe, and on CLIP it is below zero-shot on every domain. Most of the loss is on sketch. Our explanation is that when all the weights are trained, the features become specialised on the three training styles and part of the pretrained robustness is lost. The validation set cannot detect this: on the sketch runs the validation accuracy stays around 98-99% for every checkpoint, while the sketch accuracy changes by about 10 points from one checkpoint to the next. The ResNet-50 baseline shows the same behaviour.
- LoRA does not improve DINOv2. It gains on cartoon but loses on art and sketch. It fits the training domains almost perfectly, so the extra capacity is used to adapt to the training styles rather than to generalise. On CLIP, the small gain of LoRA over the linear probe (+0.4) is within the noise of a single seed.
- Training on PACS adds almost nothing to CLIP zero-shot (97.1 vs 97.2). It helps a little only on sketch. CLIP already handles the change of style well, and training on three styles mostly risks losing part of this.
- Calibration: the CLIP linear probe is under-confident (its confidence is much lower than its accuracy), probably because CLIP features give small logits with the default learning rate. All methods that train the backbone are over-confident on sketch and often confuse animals with the class horse (for example, DINOv2 with full fine-tuning labels about 160 giraffes and 150 dogs as horses).

Limitations:

- Only one seed and one data split.
- No hyperparameter search: another learning rate could change the ranking between LoRA and full fine-tuning. The linear probes often pick one of the last checkpoints, so they could benefit from more steps.
- The budgets are comparable but not identical (2000 steps for the linear probe, 1500 for LoRA and full fine-tuning, 5000 for the baseline).
- CLIP zero-shot is evaluated on the whole held-out domain, the trained models on 80% of it, which adds some noise. A version of the prompts that mentions the styles (painting, cartoon, sketch) reaches 97.6, but it uses information about the test styles, so we do not count it as a result.
- PACS is small (7 classes, about 10k images) and the new backbones are close to 100% on photo, so the conclusions may not hold on larger datasets.

Difficulties:

- DomainBed does not support CLIP or LoRA, so each notebook patches it. For CLIP the LoRA adapters had to be added in a different way, because its attention layer cannot be wrapped directly. A short test at the start checks that only the expected parameters are trained.
- The Google Drive download of PACS used by DomainBed failed, so we used the Hugging Face copy.
- Colab sessions disconnect, so every run is saved to Google Drive as soon as it finishes and is skipped if the notebook is run again.
- A first version of the LoRA notebook put the adapters in different layers for DINOv2 and CLIP, so the two were not comparable. We discarded it.

## How to run

Everything runs on Google Colab with a GPU (we used an A100). Each notebook clones DomainBed, uses the PyTorch already installed on Colab and installs only the missing packages (wilds, open_clip_torch, datasets), downloads PACS from Hugging Face, patches DomainBed and trains. There is one notebook per backbone and method, each with 4 runs, and one notebook for the final table. The baseline is the ERM notebook of Milestone 1.

| Notebook | What it does | Time on an A100 |
|---|---|---|
| `M2_run_lp_dinov2.ipynb` | DINOv2, linear probe | about 30 min |
| `M2_run_lp_clip.ipynb` | CLIP, linear probe | about 30 min |
| `M2_run_lora_dinov2.ipynb` | DINOv2, LoRA | about 40 min |
| `M2_run_lora_clip.ipynb` | CLIP, LoRA | about 30 min |
| `M2_run_fullft_dinov2.ipynb` | DINOv2, full fine-tuning | about 40 min |
| `M2_run_fullft_clip.ipynb` | CLIP, full fine-tuning | about 30 min |
| `M2_results.ipynb` | final table, error analysis, CLIP zero-shot | about 10 min |

In total, about 3-4 hours of GPU time.

1. Open a run notebook in Colab with a GPU, set SESSION_BUDGET_MIN in section 4 to the time you have, then run all cells. The first cell asks for access to Google Drive. A run that cannot finish in the given time is not started.
2. If the session stops, run all cells again: the runs already saved on Drive are skipped.
3. When the six run notebooks are done, run `M2_results.ipynb`. It reads all runs from Drive and builds the final table.

Every finished run is saved on Google Drive (`MyDrive/DeepLearningProject/M2_backup/<run>/`) with its result and the DomainBed logs. Model weights are not saved; set SAVE_CHECKPOINTS = True in section 4 to save them.

The code also supports a DINOv2 variant that uses only the class token as feature (dinov2_cls). It was planned as an optional extra experiment, but we did not run it, so it does not appear in the results.

## Files

| File | Content |
|---|---|
| `M2_run_lp_*.ipynb`, `M2_run_lora_*.ipynb`, `M2_run_fullft_*.ipynb` | the six run notebooks, with outputs |
| `M2_results.ipynb` | final table, error analysis, CLIP zero-shot |
| `results/m2_runs.jsonl` | the results of the 24 runs |
| `results/m2_summary.csv` | the results table |
