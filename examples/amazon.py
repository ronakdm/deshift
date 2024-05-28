import numpy as np
import torch.nn as nn
import torch
from tqdm import tqdm
import datetime
import time
import random
import os
import sys
import json

from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset

ROOT_DIR = "/mnt/ssd/ronak/datasets/wilds"
CACHE_DIR = "/mnt/ssd/ronak/models"
SAVE_DIR = "/mnt/hdd/ronak/wilds/amazon"
OUT_DIR = "/mnt/ssd/ronak/output/wilds/amazon"

## hyperparameters

LEARNING_RATE = 2e-5
ADAMW_TOLERANCE = 1e-8
BATCH_SIZE = 128
EPOCHS = 2
SEED = 123
UNFROZEN = 1
device = "cuda:0"

## utils

def to_dict_of_lists(lst):
    return {key: [i[key] for i in lst] for key in lst[0]}

def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

## model

n_labels = 5
model = AutoModelForSequenceClassification.from_pretrained("timinar/baby-llama-58m", cache_dir=CACHE_DIR)

# model surgery
model.score = nn.Linear(in_features=512, out_features=n_labels, bias=False)
model.num_labels = n_labels


for param in model.parameters():
    param.requires_grad = False
for module in model.model.layers[-UNFROZEN:] + [model.model.norm, model.score]:
    for param in module.parameters():
        param.requires_grad = True

model.to(device)

## data

class Amazon(Dataset):
    def __init__(self, input_ids, attn_masks, labels, metadata):
        self.input_ids = input_ids
        self.attn_masks = attn_masks
        self.labels = labels
        self.metadata = metadata

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return self.input_ids[i], self.attn_masks[i], self.labels[i], self.metadata[i]
    
def load_dataset(split):
    input_ids = torch.load(os.path.join(SAVE_DIR, f"{split}_input_ids.pt"))
    attn_masks = torch.load(os.path.join(SAVE_DIR, f"{split}_attn_masks.pt"))
    labels = torch.load(os.path.join(SAVE_DIR, f"{split}_labels.pt"))
    metadata = torch.load(os.path.join(SAVE_DIR, f"{split}_metadata.pt"))

    print(f"loaded {split} dataset: {input_ids.shape}")

    return Amazon(input_ids, attn_masks, labels, metadata)

# small enough to fit in CPU memory
train_dataset = load_dataset("train")
val_dataset = load_dataset("val")

a = train_dataset.input_ids
print(f"Train dataset size: {a.element_size() * a.numel() / (1024 ** 3):0.2f}G")

train_dataloader = DataLoader(
    train_dataset, sampler=RandomSampler(train_dataset), batch_size=BATCH_SIZE
)
print("{:>5,} training samples.".format(len(train_dataset)))
validation_dataloader = DataLoader(
    val_dataset, sampler=SequentialSampler(val_dataset), batch_size=BATCH_SIZE
)
print("{:>5,} validation samples.".format(len(val_dataset)))

## training loop

optimizer = torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE, eps = ADAMW_TOLERANCE)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = EPOCHS * BATCH_SIZE * len(train_dataloader))

# Seed everything.
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

training_stats = []
total_t0 = time.time()
for epoch_i in range(EPOCHS):

    # ========================================
    #               Training
    # ========================================

    print("")
    print("======== Epoch {:} / {:} ========".format(epoch_i + 1, EPOCHS))
    print("Training...")

    t0 = time.time()
    total_train_loss = 0

    model.train()

    for step, batch in enumerate(train_dataloader):

        if step % 200 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print(
                "  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.".format(
                    step, len(train_dataloader), elapsed
                )
            )

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()

        output = model(
            input_ids=b_input_ids,
            attention_mask=b_input_mask,
            labels=b_labels,
            return_dict=True,
        )

        loss = output.loss
        logits = output.logits
        

        total_train_loss += loss.item()

        loss.backward()

        # TODO: See if this is needed.
        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))

    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    total_eval_accuracy = 0
    total_eval_loss = 0

    for batch in validation_dataloader:

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            output = model(
                input_ids=b_input_ids,
                attention_mask=b_input_mask,
                labels=b_labels,
            )
            loss = output.loss
            logits = output.logits

        total_eval_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to("cpu").numpy()

        total_eval_accuracy += flat_accuracy(logits, label_ids)

    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    validation_time = format_time(time.time() - t0)
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    training_stats.append(
        {
            "epoch": epoch_i + 1,
            "Training Loss": avg_train_loss,
            "Valid. Loss": avg_val_loss,
            "Valid. Accur.": avg_val_accuracy,
            "Training Time": training_time,
            "Validation Time": validation_time,
        }
    )

print("")
print("Training complete!")

print(
    "Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0))
)

# Save the model.
torch.save(model, os.path.join(OUT_DIR, "erm.pt"))
training_stats = to_dict_of_lists(training_stats)
with open(os.path.join(OUT_DIR, "erm.json"), "w") as f:
    json.dump(training_stats, f, indent=2)