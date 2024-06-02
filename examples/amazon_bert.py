import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from tqdm import tqdm
import datetime
import time
import random
import os
import sys
import json
import logging

import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoTokenizer, BertForSequenceClassification, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler, SequentialSampler, Dataset

from sklearn.metrics import accuracy_score, classification_report

import sys
sys.path.extend(["..", "."])
from deshift import make_spectral_risk_measure, make_superquantile_spectrum

OBJECTIVE = sys.argv[1]
DEVICE = int(sys.argv[2])

assert OBJECTIVE in ["erm", "superquantile"]
assert DEVICE in [0, 1, 2, 3]

ROOT_DIR = "/mnt/ssd/ronak/datasets/wilds"
CACHE_DIR = "/mnt/ssd/ronak/models"
SAVE_DIR = "/mnt/hdd/ronak/wilds/amazon"
OUT_DIR = "/mnt/ssd/ronak/output/wilds/amazon"
MODEL_NAME = "bert"

LEARNING_RATE = 2e-5
ADAMW_TOLERANCE = 1e-8
BATCH_SIZE = 32
EPOCHS = 3
SEED = 123
BALANCE = 0.3 # data scaling

def load_data(split):
    input_ids  = torch.load(os.path.join(SAVE_DIR, f"{MODEL_NAME}_{split}_input_ids.pt"))
    attn_masks = torch.load(os.path.join(SAVE_DIR, f"{MODEL_NAME}_{split}_attn_masks.pt"))
    labels     = torch.load(os.path.join(SAVE_DIR, f"{MODEL_NAME}_{split}_labels.pt"))
    metadata   = torch.load(os.path.join(SAVE_DIR, f"{MODEL_NAME}_{split}_metadata.pt"))
    return input_ids, attn_masks, labels, metadata

class MaskedSequenceClassificationDataset(Dataset):
    def __init__(self, input_ids, attn_masks, labels, metadata):
        self.input_ids = input_ids
        self.attn_masks = attn_masks
        self.labels = labels
        self.metadata = metadata

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return self.input_ids[i], self.attn_masks[i], self.labels[i], self.metadata[i]
    
input_ids, attn_masks, labels, metadata = load_data("train")
train_dataset = MaskedSequenceClassificationDataset(input_ids, attn_masks, labels, metadata)

print(f"Original label distribution:  {np.bincount(train_dataset.labels.numpy()) / len(train_dataset)}")

label_dist = np.bincount(train_dataset.labels.numpy()) / len(train_dataset)
n_labels = len(label_dist)
rebalanced_dist = BALANCE * np.ones(shape=(n_labels,)) / n_labels + (1 - BALANCE) * label_dist
print(f"Rebalanced label distribution: {rebalanced_dist}")
# radon-nykodym derivative to go from unbalanced to balanced wieghts
rnd = rebalanced_dist / label_dist
sample_weight = rnd[train_dataset.labels.numpy()]

# use a weighted sampler for upsampling
# we can use more data in the forward pass with the same memory budget
n_samples = len(sample_weight) if OBJECTIVE == "erm" else 2 * len(sample_weight)
batch_size = BATCH_SIZE if OBJECTIVE == "erm" else 2 * BATCH_SIZE
train_dataloader = DataLoader(
    train_dataset, sampler=WeightedRandomSampler(sample_weight, n_samples, replacement=True), batch_size=batch_size, drop_last=True
)
# train_dataloader = DataLoader(
#     train_dataset, sampler=RandomSampler(train_dataset), batch_size=BATCH_SIZE
# )
print("{:>5,} training samples.".format(len(train_dataset)))

# create function which computes weight on each example
shift_cost = 0.05
penalty = "chi2" # options: 'chi2', 'kl'
spectrum = make_superquantile_spectrum(batch_size, 0.5)
compute_sample_weight = make_spectral_risk_measure(spectrum, penalty=penalty, shift_cost=shift_cost)

input_ids, attn_masks, labels, metadata = load_data("val")
val_dataset = MaskedSequenceClassificationDataset(input_ids, attn_masks, labels, metadata)
print(np.bincount(val_dataset.labels.numpy()) / len(val_dataset))
validation_dataloader = DataLoader(
    val_dataset, sampler=SequentialSampler(val_dataset), batch_size=BATCH_SIZE
)
print("{:>5,} validation samples.".format(len(val_dataset)))

input_ids, attn_masks, labels, metadata = load_data("test")
test_dataset = MaskedSequenceClassificationDataset(input_ids, attn_masks, labels, metadata)
test_dataloader = DataLoader(
    test_dataset, sampler=RandomSampler(test_dataset), batch_size=BATCH_SIZE
)
print("{:>5,} test samples.".format(len(test_dataset)))


model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=5,
    output_attentions=False,
    output_hidden_states=False,
    cache_dir=CACHE_DIR,
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE, eps = ADAMW_TOLERANCE)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = EPOCHS * BATCH_SIZE * len(train_dataloader))

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(OUT_DIR, "output_{MODEL_NAME}_{OBJECTIVE}.log")),
        logging.FileHandler("output_{MODEL_NAME}_{OBJECTIVE}.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

logging.info(f"****************************************************************")
logging.info(f"TRAINING AMAZON BERT with {OBJECTIVE} LOSS on DEVICE {DEVICE}...")
logging.info(f"****************************************************************")


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
    logging.info("======== Epoch {:} / {:} ========".format(epoch_i + 1, EPOCHS))
    logging.info("Training...")

    t0 = time.time()
    total_train_loss = 0
    total_train_objective = 0
    total_train_accuracy = 0

    model.train()

    for step, batch in enumerate(train_dataloader):

        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            logging.info(
                "  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.   Loss: {:0.5f}".format(
                    step, len(train_dataloader), elapsed, total_train_loss / step
                )
            )

        b_input_ids = batch[0].to(DEVICE)
        b_input_mask = batch[1].to(DEVICE)
        b_labels = batch[2].to(DEVICE)

        model.zero_grad()

        if OBJECTIVE == "superquantile":
            # use only the examples with high loss.
            with torch.no_grad():
                output = model(
                    input_ids=b_input_ids,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                    return_dict=True,
                )
                logits = output.logits
                losses = F.cross_entropy(logits, b_labels, reduction="none")
                q = compute_sample_weight(losses)
                # sort, argsort = torch.sort(losses, stable=True)
                # rank = torch.argsort(argsort)
        
                b_input_ids = b_input_ids[q > 0]
                b_input_mask = b_input_mask[q > 0]
                b_labels = b_labels[q > 0]
                weights = q[q > 0]

        output = model(
            input_ids=b_input_ids,
            attention_mask=b_input_mask,
            labels=b_labels,
            return_dict=True,
        )

        logits = output.logits

        # sample weighted loss
        if OBJECTIVE == "superquantile":
            losses = F.cross_entropy(logits, b_labels, reduction="none")
            loss = weights @ losses
        else:
            loss = output.loss

        total_train_loss += output.loss.item()
        total_train_objective += loss.item()
        loss.backward()
        

        # TODO: See if this is needed.
        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        total_train_accuracy += flat_accuracy(logits.detach().cpu().numpy(), b_labels.detach().cpu().numpy())

    avg_train_loss = total_train_loss / len(train_dataloader)
    avg_train_objective = total_train_objective / len(train_dataloader)
    avg_train_accuracy = total_train_accuracy / len(train_dataloader)
    training_time = format_time(time.time() - t0)

    print("")
    logging.info("  Average training loss:      {0:.3f}".format(avg_train_objective))
    logging.info("  Average training objective: {0:.3f}".format(avg_train_objective))
    logging.info("  Average training accuracy:  {0:.3f}".format(avg_train_accuracy))
    logging.info("  Training epoch took:        {:}".format(training_time))

    # ========================================
    #               Validation
    # ========================================

    print("")
    logging.info("Running Validation...")

    t0 = time.time()

    model.eval()

    total_eval_accuracy = 0
    total_eval_loss = 0

    y_true = []
    y_pred = []
    for batch in validation_dataloader:

        b_input_ids = batch[0].to(DEVICE)
        b_input_mask = batch[1].to(DEVICE)
        b_labels = batch[2].to(DEVICE)

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

        y_pred.append(np.argmax(logits, axis=1))
        y_true.append(label_ids)

        # total_eval_accuracy += flat_accuracy(logits, label_ids)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    avg_val_accuracy = accuracy_score(y_true, y_pred)
    logging.info("  Validation Accuracy: {0:.3f}".format(avg_val_accuracy))
    logging.info(classification_report(y_true, y_pred, zero_division=0.0))
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    validation_time = format_time(time.time() - t0)
    logging.info("  Validation Loss: {0:.3f}".format(avg_val_loss))
    logging.info("  Validation took: {:}".format(validation_time))

    
    epoch_stats = {
        "epoch": epoch_i + 1,
        "train_loss": avg_train_loss,
        "train_acc": avg_train_accuracy,
        "val_loss": avg_val_loss,
        "val_acc": avg_val_accuracy,
        "train_time": training_time,
        "val_time": validation_time,
        "val_report": classification_report(y_true, y_pred, zero_division=0.0, output_dict=True)
    }
    with open(os.path.join(OUT_DIR, f"{MODEL_NAME}_{OBJECTIVE}_epoch_{epoch_i}.json"), "w") as f:
        json.dump(epoch_stats, f, indent=2)
    training_stats.append(epoch_stats)
    torch.save(model.state_dict(), os.path.join(OUT_DIR, f"{MODEL_NAME}_{OBJECTIVE}_epoch_{epoch_i}.pt"))

print("")
logging.info("Training complete!")
logging.info(
    "Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0))
)

# Save the model.
torch.save(model.state_dict(), os.path.join(OUT_DIR, f"{MODEL_NAME}_{OBJECTIVE}.pt"))
training_stats = to_dict_of_lists(training_stats)
with open(os.path.join(OUT_DIR, f"{MODEL_NAME}_{OBJECTIVE}_training_stats.json"), "w") as f:
    json.dump(training_stats, f, indent=2)