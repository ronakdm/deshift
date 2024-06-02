
import torch
import os

def load_data(split):
    input_ids  = torch.load(os.path.join(SAVE_DIR, f"{MODEL_NAME}_{split}_input_ids.pt"))
    attn_masks = torch.load(os.path.join(SAVE_DIR, f"{MODEL_NAME}_{split}_attn_masks.pt"))
    labels     = torch.load(os.path.join(SAVE_DIR, f"{MODEL_NAME}_{split}_labels.pt"))
    metadata   = torch.load(os.path.join(SAVE_DIR, f"{MODEL_NAME}_{split}_metadata.pt"))
    return input_ids, attn_masks, labels, metadata

class MaskedSequenceClassificationDataset(torch.utils.Dataset):
    def __init__(self, input_ids, attn_masks, labels, metadata):
        self.input_ids = input_ids
        self.attn_masks = attn_masks
        self.labels = labels
        self.metadata = metadata

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return self.input_ids[i], self.attn_masks[i], self.labels[i], self.metadata[i]
    

