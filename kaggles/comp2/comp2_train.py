
import os
import json
import random
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(".")

# ------------------------------------------------------------
# MODEL SELECTION
# ------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Train ConvNeXt model: tiny / small / base"
)

parser.add_argument(
    "--model",
    choices=["tiny", "small", "base"],
    default="tiny",
    help="Выбор размера ConvNeXt"
)

args, _ = parser.parse_known_args()

MODEL_SIZE = args.model

MODEL_CONFIGS = {
    "tiny": {
        "name": "convnext_tiny.in12k_ft_in1k",
        "batch_size": 64,
        "lr": 3e-4,
    },
    "small": {
        "name": "convnext_small.in12k_ft_in1k",
        "batch_size": 32,
        "lr": 1.5e-4,
    },
    "base": {
        "name": "convnext_base.in12k_ft_in1k",
        "batch_size": 16,
        "lr": 1e-4,
    },
}

MODEL_CFG = MODEL_CONFIGS[MODEL_SIZE]

MODEL_NAME = MODEL_CFG["name"]
BS = MODEL_CFG["batch_size"]
MODEL_LR = MODEL_CFG["lr"]

MODEL_TAG = MODEL_NAME.replace(".", "_")


# ------------------------------------------------------------
# IMAGE AND TRAINING SETTINGS
# ------------------------------------------------------------

IMG = 192
TEST_IMG = 288

FULL_TRAIN = True
BALANCED_SAMPLING = True

VAL_FRAC = 0.15
SEED = 0

# Общее количество эпох.
# После этой эпохи обучение автоматически завершится.
TRAIN_EPOCHS = 70

HEAD_LR_MULTIPLIER = 10.0


# ------------------------------------------------------------
# ONE CYCLE LR SETTINGS
# ------------------------------------------------------------

PCT_START = 0.15
DIV_FACTOR = 25.0
FINAL_DIV_FACTOR = 10000.0


# ------------------------------------------------------------
# MODEL-SPECIFIC FILES
# ------------------------------------------------------------

LAST_CKPT = f"model_last_{MODEL_TAG}_training_state.pt"
BEST_CKPT = f"model_best_{MODEL_TAG}.pt"

MODEL_EXPORT = f"model_{MODEL_TAG}.pt"

SUB_CSV = f"submission_out_{MODEL_TAG}.csv"
BEST_SUB_CSV = f"submission_best_{MODEL_TAG}.csv"


# ------------------------------------------------------------
# TRAINING CHECKPOINT SETTINGS
# ------------------------------------------------------------

RESUME_TRAINING = True


# ============================================================
# DEVICE AND RANDOM SEEDS
# ============================================================

dev = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_AMP = dev.type == "cuda"

torch.backends.cudnn.benchmark = True

print("\n" + "=" * 65)
print("MODEL:", MODEL_SIZE)
print("TIMM ARCHITECTURE:", MODEL_NAME)
print("BATCH SIZE:", BS)
print("BACKBONE LR:", MODEL_LR)
print("TOTAL EPOCHS:", TRAIN_EPOCHS)
print("DEVICE:", dev)

if USE_AMP:
    print("GPU:", torch.cuda.get_device_name(0))

print("=" * 65)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if USE_AMP:
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# LABELS
# ============================================================

with open(ROOT / "labels.json", "r") as f:
    labels = json.load(f)

NUM_CLASSES = len(labels)

name2idx = {
    v: int(k)
    for k, v in labels.items()
}

EXTS = {
    ".jpg", ".jpeg", ".png"
}


# ============================================================
# TRAINING DATA
# ============================================================

rows = []

for cls_dir in sorted(
    (ROOT / "data" / "train").iterdir()
):

    if not cls_dir.is_dir():
        continue

    if cls_dir.name not in name2idx:
        print("Skipping unknown folder:", cls_dir.name)
        continue

    for f in sorted(cls_dir.iterdir()):

        if f.suffix.lower() in EXTS:

            rows.append({
                "path": str(f),
                "label": name2idx[cls_dir.name]
            })


ann = pd.DataFrame(rows)

if ann.empty:
    raise RuntimeError("No training images found.")

print("\nTraining images:", len(ann))


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

rng = np.random.RandomState(SEED)

val_indices = []

for _, group in ann.groupby("label"):

    n_val = (
        min(
            len(group) - 1,
            max(1, int(round(len(group) * VAL_FRAC)))
        )
        if len(group) >= 2
        else 0
    )

    if n_val:

        val_indices.extend(
            rng.choice(
                group.index.to_numpy(),
                size=n_val,
                replace=False
            ).tolist()
        )


val_idx = pd.Index(val_indices)

if FULL_TRAIN:

    tr_df = ann.reset_index(drop=True)
    va_df = ann.iloc[0:0].copy()

else:

    tr_df = ann.drop(val_idx).reset_index(drop=True)
    va_df = ann.loc[val_idx].reset_index(drop=True)

    if len(va_df) == 0:
        raise ValueError("Validation split is empty.")


# ============================================================
# TRANSFORMS
# ============================================================

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

train_tf = T.Compose([

    T.RandomResizedCrop(
        IMG,
        scale=(0.45, 1.0),
        ratio=(0.8, 1.25)
    ),

    T.RandomHorizontalFlip(),

    T.TrivialAugmentWide(),

    T.RandomRotation(8),

    T.ColorJitter(
        0.18,
        0.18,
        0.18,
        0.02
    ),

    T.RandomGrayscale(0.02),

    T.ToTensor(),

    T.Normalize(MEAN, STD),

    T.RandomErasing(
        p=0.15,
        scale=(0.02, 0.10)
    )
])


eval_tf = T.Compose([

    T.Resize((TEST_IMG, TEST_IMG)),

    T.ToTensor(),

    T.Normalize(MEAN, STD)
])


# ============================================================
# DATASET
# ============================================================

class DS(Dataset):

    def __init__(self, paths, ys, tf):

        self.paths = list(paths)
        self.ys = list(ys)
        self.tf = tf

        self.imgs = []

        for p in self.paths:

            with Image.open(p) as im:

                if im.mode == "P":
                    im = im.convert("RGBA")

                im = im.convert("RGB")

                im.thumbnail((
                    max(IMG, TEST_IMG) * 2,
                    max(IMG, TEST_IMG) * 2
                ))

                self.imgs.append(im.copy())

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):

        return (
            self.tf(self.imgs[i]),
            int(self.ys[i])
        )


tr_ds = DS(
    tr_df["path"],
    tr_df["label"],
    train_tf
)

va_ds = DS(
    va_df["path"],
    va_df["label"],
    eval_tf
)


# ============================================================
# DATALOADERS
# ============================================================

if BALANCED_SAMPLING:

    counts = tr_df["label"].value_counts().to_dict()

    weights = torch.DoubleTensor([

        1.0 / counts[int(y)]

        for y in tr_df["label"]

    ])

    sampler = torch.utils.data.WeightedRandomSampler(
        weights,
        len(weights),
        replacement=True
    )

    tr_dl = DataLoader(
        tr_ds,
        batch_size=BS,
        sampler=sampler,
        num_workers=0,
        pin_memory=USE_AMP
    )

else:

    tr_dl = DataLoader(
        tr_ds,
        batch_size=BS,
        shuffle=True,
        num_workers=0,
        pin_memory=USE_AMP
    )


va_dl = (

    DataLoader(
        va_ds,
        batch_size=BS,
        num_workers=0
    )

    if not FULL_TRAIN

    else None
)


# ============================================================
# MIXUP / CUTMIX
# ============================================================

def mix_batch(x, y):

    r = random.random()

    if r < 0.5:

        return x, y, y, 1.0

    idx = torch.randperm(
        x.size(0),
        device=x.device
    )

    if r < 0.75:

        lam = float(
            np.random.beta(0.3, 0.3)
        )

        x = (
            lam * x
            + (1 - lam) * x[idx]
        )

        return x, y, y[idx], lam

    lam = float(
        np.random.beta(1.0, 1.0)
    )

    H, W = x.shape[2], x.shape[3]

    ch = int(H * np.sqrt(1 - lam))
    cw = int(W * np.sqrt(1 - lam))

    cy = np.random.randint(H)
    cx = np.random.randint(W)

    y1 = max(cy - ch // 2, 0)
    y2 = min(cy + ch // 2, H)

    x1 = max(cx - cw // 2, 0)
    x2 = min(cx + cw // 2, W)

    x = x.clone()

    x[:, :, y1:y2, x1:x2] = (
        x[idx][:, :, y1:y2, x1:x2]
    )

    lam = 1 - (
        (y2 - y1) * (x2 - x1) / (H * W)
    )

    return x, y, y[idx], lam

# ============================================================
# METRICS
# ============================================================

def macro_f1(y_true, y_pred, n=NUM_CLASSES):

    f1s = []

    for c in range(n):

        tp = np.sum(
            (y_true == c) & (y_pred == c)
        )

        fp = np.sum(
            (y_true != c) & (y_pred == c)
        )

        fn = np.sum(
            (y_true == c) & (y_pred != c)
        )

        denominator = 2 * tp + fp + fn

        f1 = (
            0.0
            if denominator == 0
            else 2 * tp / denominator
        )

        f1s.append(f1)

    return float(np.mean(f1s)), f1s


def accuracy_score(y_true, y_pred):

    return float(
        np.mean(y_true == y_pred)
    )


# ============================================================
# MODEL
# ============================================================

print("\nCreating model:", MODEL_NAME)

net = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=NUM_CLASSES,
    drop_path_rate=0.1
).to(dev)


# Separate classifier and backbone parameters.

head_ids = {
    id(p)
    for p in net.get_classifier().parameters()
}

head_params = [
    p for p in net.parameters()
    if id(p) in head_ids
]

body_params = [
    p for p in net.parameters()
    if id(p) not in head_ids
]


opt = torch.optim.AdamW(

    [
        {
            "params": body_params,
            "lr": MODEL_LR / DIV_FACTOR,
            "max_lr": MODEL_LR
        },

        {
            "params": head_params,
            "lr": (
                MODEL_LR
                * HEAD_LR_MULTIPLIER
                / DIV_FACTOR
            ),
            "max_lr": (
                MODEL_LR
                * HEAD_LR_MULTIPLIER
            )
        }

    ],

    weight_decay=0.05

)


scaler = torch.amp.GradScaler(
    "cuda",
    enabled=USE_AMP
)


criterion = nn.CrossEntropyLoss(
    label_smoothing=0.1
)


# ============================================================
# ONE CYCLE LR
# ============================================================

def create_scheduler():

    return torch.optim.lr_scheduler.OneCycleLR(

        opt,

        max_lr=[
            MODEL_LR,
            MODEL_LR * HEAD_LR_MULTIPLIER
        ],

        epochs=TRAIN_EPOCHS,

        steps_per_epoch=len(tr_dl),

        pct_start=PCT_START,

        anneal_strategy="cos",

        div_factor=DIV_FACTOR,

        final_div_factor=FINAL_DIV_FACTOR

    )


# ============================================================
# RESUME TRAINING
# ============================================================

start_epoch = 1

scheduler = create_scheduler()


if RESUME_TRAINING and os.path.exists(LAST_CKPT):

    print("\\nResuming latest training state...")

    checkpoint = torch.load(
        LAST_CKPT,
        map_location=dev,
        weights_only=False
    )

    saved_model_name = checkpoint.get("model_name")

    if (
        saved_model_name is not None
        and saved_model_name != MODEL_NAME
    ):
        raise RuntimeError(
            f"Checkpoint contains {saved_model_name}, "
            f"but selected model is {MODEL_NAME}."
        )

    if checkpoint.get("training_version") != 2:
        raise RuntimeError(
            "This checkpoint was created by an incompatible script version. "
            "Rename or remove LAST_CKPT to start a fresh run."
        )

    expected_steps = TRAIN_EPOCHS * len(tr_dl)
    saved_scheduler = checkpoint.get("scheduler", {})

    if saved_scheduler.get("total_steps") != expected_steps:
        raise RuntimeError(
            "The epoch count or DataLoader length changed. "
            "Use the original settings to resume, or start a new run."
        )

    net.load_state_dict(checkpoint["model"])
    opt.load_state_dict(checkpoint["optimizer"])

    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    start_epoch = checkpoint["epoch"] + 1
    scheduler.load_state_dict(checkpoint["scheduler"])

    print("Resuming at epoch:", start_epoch)


# ============================================================
# FINITE TRAINING
# ============================================================

print("\nStarting training...")
print("Model:", MODEL_NAME)
print("Total epochs:", TRAIN_EPOCHS)
print("Starting epoch:", start_epoch)
print("Latest checkpoint:", LAST_CKPT)
print("Model export:", MODEL_EXPORT)

completed_epoch = start_epoch - 1

if start_epoch > TRAIN_EPOCHS:
    print("\nTraining already completed!")
    print("Last saved epoch:", completed_epoch)

else:
    try:
        for epoch in range(start_epoch, TRAIN_EPOCHS + 1):
            net.train()
            t0 = time.time()
            total_loss = 0.0

            # ====================================================
            # TRAIN
            # ====================================================

            for batch_idx, (x, y) in enumerate(tr_dl):
                x = x.to(dev)
                y = y.to(dev)

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=USE_AMP
                ):
                    x, ya, yb, lam = mix_batch(x, y)
                    out = net(x)
                    loss = (
                        lam * criterion(out, ya)
                        + (1 - lam) * criterion(out, yb)
                    )

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                # OneCycleLR: one step per batch.
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(tr_dl)

            # ====================================================
            # SAVE LATEST TRAINING STATE
            # ====================================================

            torch.save(
                {
                    "training_version": 2,
                    "epoch": epoch,
                    "model_size": MODEL_SIZE,
                    "model_name": MODEL_NAME,
                    "model": net.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                },
                LAST_CKPT
            )

            # Save a directly loadable state_dict for the ensemble.
            torch.save(net.state_dict(), MODEL_EXPORT)

            # ====================================================
            # DISPLAY TRAINING RESULTS
            # ====================================================

            elapsed = time.time() - t0
            current_lrs = [
                group["lr"]
                for group in opt.param_groups
            ]
            completed_epoch = epoch

            print("\n" + "=" * 65)
            print(
                f"Model: {MODEL_SIZE} "
                f"| Epoch {epoch}/{TRAIN_EPOCHS}"
            )
            print(f"Loss: {avg_loss:.5f}")
            print(f"Backbone LR: {current_lrs[0]:.10f}")
            print(f"Classifier LR: {current_lrs[1]:.10f}")
            print(f"Time: {elapsed:.1f}s")
            print("Training checkpoint:", LAST_CKPT)
            print("Model weights:", MODEL_EXPORT)
            print("=" * 65)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        # Preserve the latest model weights even after interruption.
        torch.save(net.state_dict(), MODEL_EXPORT)
        print("Training checkpoint:", LAST_CKPT)
        print("Model weights:", MODEL_EXPORT)


# ============================================================
# FINISH
# ============================================================

# Ensure the final .pt file exists and contains only model weights.
torch.save(net.state_dict(), MODEL_EXPORT)

print("\n" + "=" * 65)
print("TRAINING FINISHED")
print("Model:", MODEL_NAME)
print(f"Completed epochs: {completed_epoch}/{TRAIN_EPOCHS}")
print("Latest training checkpoint:", LAST_CKPT)
print("Model weights:", MODEL_EXPORT)
print("=" * 65)
