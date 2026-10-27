"""Milestone 1: data acquisition, exploration and preparation for PACS.

Steps (each one is a function, so the notebook can call them one by one):
  1. download_pacs   - download PACS from Hugging Face (flwrlabs/pacs) and save it
                       as <data_dir>/PACS/<domain>/<class>/<id>.jpg (DomainBed layout)
  2. build_index     - one row per image: path, domain, class, label, size, pixel stats
  3. explore         - tables and figures (class distribution, samples, pixel statistics,
                       duplicates)
  4. make_splits     - DomainBed's split (20% of every domain held out for validation,
                       same seeds as DomainBed) and the 4 leave-one-domain-out settings
  5. load_split      - turns a split CSV into model inputs X (N, 3, 224, 224) and
                       outputs y (N,)

Usage (in a Colab cell, from the repository folder):
  !python prepare_data.py                      # everything, data in ./data
  !python prepare_data.py --skip-figures       # no figures and stats, only download and splits
"""

import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd
from PIL import Image

HF_DATASET = "flwrlabs/pacs"
DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]   # sorted, as in DomainBed
N_IMAGES = 9991
HOLDOUT_FRACTION = 0.2       # DomainBed --holdout_fraction default
TRIAL_SEED = 0               # DomainBed --trial_seed default
IMAGE_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# one color per domain, the same in every figure
DOMAIN_COLORS = {"art_painting": "#2a78d6", "cartoon": "#eb6834",
                 "photo": "#1baf7a", "sketch": "#eda100"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# --------------------------------------------------------------------------- 1. download
def download_pacs(data_dir="data"):
    """Download PACS from Hugging Face and write it in DomainBed's folder layout.

    The images are re-encoded as RGB JPEG (quality 95), so every file can be opened
    the same way. Files are numbered per (domain, class) in the order of the Hugging
    Face dataset, so the result is the same on every machine. Skipped if already done.
    """
    root = f"{data_dir}/PACS"
    if count_images(root) == N_IMAGES:
        print(f"PACS already in {root} ({N_IMAGES} images), download skipped")
        return _read_class_names(data_dir)

    from datasets import load_dataset   # only needed for the download
    ds = load_dataset(HF_DATASET, split="train")
    class_names = ds.features["label"].names
    counts = {}
    for i, ex in enumerate(ds):
        domain, cls = ex["domain"], class_names[ex["label"]]
        out_dir = os.path.join(root, domain, cls)
        os.makedirs(out_dir, exist_ok=True)
        idx = counts.get((domain, cls), 0)
        counts[(domain, cls)] = idx + 1
        ex["image"].convert("RGB").save(os.path.join(out_dir, f"{idx:05d}.jpg"),
                                        "JPEG", quality=95)
        if i % 2000 == 0:
            print(f"written {i}/{len(ds)}", flush=True)

    with open(os.path.join(data_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=1)
    print(f"PACS ready in {root}: {count_images(root)} images")
    return class_names


def count_images(root):
    if not os.path.isdir(root):
        return 0
    return sum(len(files) for _, _, files in os.walk(root))


def _read_class_names(data_dir):
    path = os.path.join(data_dir, "class_names.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return sorted(os.listdir(os.path.join(data_dir, "PACS", DOMAINS[0])))


# --------------------------------------------------------------------------- 2. index
def build_index(data_dir="data"):
    """One row per image, in the order DomainBed's ImageFolder reads them
    (domains sorted, classes sorted, files sorted). Saved to data/pacs_index.csv."""
    class_names = _read_class_names(data_dir)
    rows = []
    for domain in DOMAINS:
        idx_in_domain = 0
        for label, cls in enumerate(class_names):
            folder = os.path.join(data_dir, "PACS", domain, cls)
            for fname in sorted(os.listdir(folder)):
                path = os.path.join(folder, fname)
                with open(path, "rb") as f:
                    raw = f.read()
                img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255
                gray = img.mean(axis=2)
                rows.append({
                    "path": f"PACS/{domain}/{cls}/{fname}",
                    "domain": domain,
                    "class_name": cls,
                    "label": label,
                    "index_in_domain": idx_in_domain,
                    "width": img.shape[1],
                    "height": img.shape[0],
                    "brightness": gray.mean(),
                    "contrast": gray.std(),
                    # mean distance of the channels from gray: ~0 for black-and-white images
                    "colorfulness": np.abs(img - gray[..., None]).mean(),
                    "md5": hashlib.md5(raw).hexdigest(),
                })
                idx_in_domain += 1
    index = pd.DataFrame(rows)
    index.to_csv(os.path.join(data_dir, "pacs_index.csv"), index=False)
    return index


# --------------------------------------------------------------------------- 3. explore
def class_table(index):
    """Number of images per domain (rows) and class (columns)."""
    t = pd.crosstab(index["domain"], index["class_name"], margins=True, margins_name="total")
    return t


def duplicate_report(index):
    """Exact duplicates (same file content). Since all images were re-encoded with the
    same settings, identical source images give identical files."""
    dup = index[index.duplicated("md5", keep=False)].copy()
    groups = dup.groupby("md5")
    rows = []
    for md5, g in groups:
        rows.append({"md5": md5, "copies": len(g),
                     "domains": ",".join(sorted(set(g["domain"]))),
                     "classes": ",".join(sorted(set(g["class_name"]))),
                     "paths": " | ".join(g["path"])})
    return pd.DataFrame(rows, columns=["md5", "copies", "domains", "classes", "paths"])


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_class_distribution(index, out_path=None):
    import matplotlib.pyplot as plt
    t = pd.crosstab(index["class_name"], index["domain"])[DOMAINS]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(t.index))
    w = 0.2
    for i, d in enumerate(DOMAINS):
        ax.bar(x + (i - 1.5) * w, t[d], width=w - 0.02, color=DOMAIN_COLORS[d],
               label=f"{d} ({t[d].sum()})")
    ax.set_xticks(x, t.index)
    ax.set_ylabel("images", color=INK_2)
    ax.set_title("PACS: images per class and domain", color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.0))
    ax.set_ylim(0, t.values.max() * 1.18)
    _style(ax)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120)
    return fig


def plot_samples(index, data_dir="data", out_path=None, seed=0):
    """One random image for every (domain, class) pair."""
    import matplotlib.pyplot as plt
    classes = sorted(index["class_name"].unique())
    fig, axes = plt.subplots(len(DOMAINS), len(classes), figsize=(1.6 * len(classes), 1.75 * len(DOMAINS)))
    rng = np.random.RandomState(seed)
    for r, d in enumerate(DOMAINS):
        for c, cls in enumerate(classes):
            sub = index[(index["domain"] == d) & (index["class_name"] == cls)]
            row = sub.iloc[rng.randint(len(sub))]
            ax = axes[r, c]
            ax.imshow(Image.open(os.path.join(data_dir, row["path"])))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(cls, fontsize=10, color=INK)
            if c == 0:
                ax.set_ylabel(d, fontsize=10, color=INK)
    fig.suptitle("Same 7 classes, 4 visual styles (domains)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120)
    return fig


def plot_pixel_stats(index, out_path=None):
    """Distribution of brightness, contrast and colorfulness per domain."""
    import matplotlib.pyplot as plt
    stats = [("brightness", "mean gray level"), ("contrast", "std of gray level"),
             ("colorfulness", "mean |channel - gray|")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (col, xlabel) in zip(axes, stats):
        bins = np.linspace(index[col].min(), index[col].max(), 40)
        for d in DOMAINS:
            v = index.loc[index["domain"] == d, col]
            ax.hist(v, bins=bins, histtype="step", linewidth=2, color=DOMAIN_COLORS[d],
                    density=True, label=d)
        ax.set_title(col, color=INK, loc="left", fontsize=11)
        ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
        ax.set_yticks([])
        _style(ax)
    axes[0].set_ylabel("density", color=INK_2)
    axes[-1].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120)
    return fig


def explore(index, data_dir="data", fig_dir="figures"):
    os.makedirs(fig_dir, exist_ok=True)
    stats_dir = os.path.join(data_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)

    ct = class_table(index)
    ct.to_csv(os.path.join(stats_dir, "class_counts.csv"))
    print("\nImages per domain and class:\n", ct.to_string())

    sizes = index.groupby(["width", "height"]).size().rename("images").reset_index()
    print("\nImage sizes (width x height):\n", sizes.to_string(index=False))

    px = index.groupby("domain")[["brightness", "contrast", "colorfulness"]].mean().round(3)
    px.to_csv(os.path.join(stats_dir, "pixel_stats.csv"))
    print("\nMean pixel statistics per domain:\n", px.to_string())

    dups = duplicate_report(index)
    dups.to_csv(os.path.join(stats_dir, "duplicates.csv"), index=False)
    print(f"\nExact duplicates: {len(dups)} groups, "
          f"{int(dups['copies'].sum() - len(dups)) if len(dups) else 0} extra copies")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for fn, name in [(plot_class_distribution, "class_distribution.png"),
                     (plot_pixel_stats, "pixel_stats.png")]:
        plt.close(fn(index, out_path=os.path.join(fig_dir, name)))
    plt.close(plot_samples(index, data_dir, out_path=os.path.join(fig_dir, "samples.png")))
    print(f"figures saved in {fig_dir}/")


# --------------------------------------------------------------------------- 4. splits
def seed_hash(*args):
    """Same as domainbed.lib.misc.seed_hash: an integer hash of the arguments."""
    return int(hashlib.md5(str(args).encode("utf-8")).hexdigest(), 16) % (2 ** 31)


def assign_parts(index, holdout_fraction=HOLDOUT_FRACTION, trial_seed=TRIAL_SEED):
    """Mark each image as 'in' (80%) or 'out' (20%) exactly as DomainBed's train.py does:
    for domain env_i, shuffle range(len(env)) with seed_hash(trial_seed, env_i) and take
    the first int(len * holdout_fraction) indices as 'out'."""
    index = index.copy()
    index["part"] = "in"
    for env_i, d in enumerate(DOMAINS):
        rows = index.index[index["domain"] == d]
        keys = list(range(len(rows)))
        np.random.RandomState(seed_hash(trial_seed, env_i)).shuffle(keys)
        n_out = int(len(rows) * holdout_fraction)
        out_pos = index.loc[rows, "index_in_domain"].isin(keys[:n_out])
        index.loc[rows[out_pos.values], "part"] = "out"
    return index


def make_splits(index, data_dir="data", holdout_fraction=HOLDOUT_FRACTION, trial_seed=TRIAL_SEED):
    """Leave-one-domain-out: for each test domain write
         train.csv - 'in' part (80%) of the 3 training domains
         val.csv   - 'out' part (20%) of the 3 training domains (model selection)
         test.csv  - all images of the held-out domain (never used for training or selection)
    to <data_dir>/splits/test_<domain>/. Returns a summary table."""
    index = assign_parts(index, holdout_fraction, trial_seed)
    cols = ["path", "domain", "class_name", "label", "part"]
    index[cols].to_csv(os.path.join(data_dir, "pacs_index_parts.csv"), index=False)
    summary = []
    for test_domain in DOMAINS:
        train_d = index["domain"] != test_domain
        splits = {
            "train": index[train_d & (index["part"] == "in")],
            "val": index[train_d & (index["part"] == "out")],
            "test": index[~train_d],
        }
        out_dir = os.path.join(data_dir, "splits", f"test_{test_domain}")
        os.makedirs(out_dir, exist_ok=True)
        for name, df in splits.items():
            df[cols].to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
        summary.append({"test_domain": test_domain,
                        **{name: len(df) for name, df in splits.items()}})
    summary = pd.DataFrame(summary)
    check_splits(index, data_dir)
    return summary


def check_splits(index, data_dir="data"):
    """Sanity checks: no image in two splits, test domain absent from train/val,
    every class present in every split, all images used."""
    for test_domain in DOMAINS:
        d = os.path.join(data_dir, "splits", f"test_{test_domain}")
        s = {n: pd.read_csv(os.path.join(d, f"{n}.csv")) for n in ("train", "val", "test")}
        paths = [set(df["path"]) for df in s.values()]
        assert not (paths[0] & paths[1] or paths[0] & paths[2] or paths[1] & paths[2]), "overlap"
        assert sum(len(p) for p in paths) == len(index), "images missing"
        assert test_domain not in set(s["train"]["domain"]) | set(s["val"]["domain"])
        assert set(s["test"]["domain"]) == {test_domain}
        for n, df in s.items():
            assert df["label"].nunique() == 7, f"{test_domain}/{n}: missing class"
    print("split checks passed: disjoint splits, held-out domain unseen, all 7 classes everywhere")


# --------------------------------------------------------------------------- 5. inputs/outputs
def preprocess(path, train=False, rng=None):
    """Image -> float32 array (3, 224, 224), normalized with the ImageNet statistics
    (the backbones are ImageNet-pretrained). With train=True a DomainBed-style
    augmentation is applied: random resized crop (scale 0.7-1), horizontal flip,
    color jitter (0.3) and random grayscale (p=0.1)."""
    img = Image.open(path).convert("RGB")
    if train:
        rng = rng or np.random
        w, h = img.size
        area = w * h * rng.uniform(0.7, 1.0)
        ratio = np.exp(rng.uniform(np.log(3 / 4), np.log(4 / 3)))
        cw = min(w, int(round(np.sqrt(area * ratio))))
        ch = min(h, int(round(np.sqrt(area / ratio))))
        x0, y0 = rng.randint(0, w - cw + 1), rng.randint(0, h - ch + 1)
        img = img.crop((x0, y0, x0 + cw, y0 + ch))
        if rng.rand() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255
    if train:
        x = x * rng.uniform(0.7, 1.3)                                   # brightness
        m = x.mean()
        x = (x - m) * rng.uniform(0.7, 1.3) + m                         # contrast
        g = x.mean(axis=2, keepdims=True)
        x = (x - g) * rng.uniform(0.7, 1.3) + g                         # saturation
        if rng.rand() < 0.1:
            x = np.repeat(x.mean(axis=2, keepdims=True), 3, axis=2)     # grayscale
        x = np.clip(x, 0, 1)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.transpose(2, 0, 1)


def load_split(test_domain, split, data_dir="data", limit=None, train_aug=False, seed=0):
    """Model inputs and outputs of one split: X (N, 3, 224, 224) float32, y (N,) int64.
    With limit=None the whole split is loaded (about 0.6 MB per image in memory)."""
    df = pd.read_csv(os.path.join(data_dir, "splits", f"test_{test_domain}", f"{split}.csv"))
    if limit is not None:
        df = df.sample(n=min(limit, len(df)), random_state=seed)
    rng = np.random.RandomState(seed)
    X = np.stack([preprocess(os.path.join(data_dir, p), train_aug, rng) for p in df["path"]])
    y = df["label"].to_numpy(dtype=np.int64)
    return X, y


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--fig-dir", default="figures")
    ap.add_argument("--skip-figures", action="store_true")
    args = ap.parse_args()

    download_pacs(args.data_dir)
    print("building index (reads every image once)...")
    index = build_index(args.data_dir)
    if not args.skip_figures:
        explore(index, args.data_dir, args.fig_dir)
    summary = make_splits(index, args.data_dir)
    print("\nLeave-one-domain-out splits (number of images):\n", summary.to_string(index=False))

    X, y = load_split("sketch", "train", args.data_dir, limit=8, train_aug=True)
    print(f"\nexample batch from test_sketch/train: X {X.shape} {X.dtype}, y {y.shape} {y.tolist()}")


if __name__ == "__main__":
    main()
