import torch
import torch.nn as nn
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Union, List
import sys
import os
import pickle
import json
import model
import generation
from model import MathTokenizer
from model import GCDTransformer
from generation import GCDDataset

# Create a directory for all outputs
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def compute_accuracy(model, dataloader, device):
    model.eval()
    perfect_sequences = 0
    total_sequences = 0
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            output = model(src, tgt_input)
            preds = output.argmax(-1)
            mask = (tgt_output != model.pad_id)
            seq_match = (preds == tgt_output) | ~mask
            perfect_sequences += seq_match.all(dim=1).sum().item()
            total_sequences += tgt.size(0)
    return perfect_sequences / max(total_sequences, 1)

# Set the bases and distributions to test
bases = [2, 3, 4, 5, 6, 7, 10, 11, 12, 16, 20, 36]
distributions = ["uniform", "normal", "loguniform", "poisson",
                 "geometric", "exponential"]

#Set model hyperparameters 

validate_step = 2
layers = 4
heads = 8
hidden_dimension = 512
length = 512
lr = 1e-4
batch = 256
max_int = 1_000_000
sample_size = 10_000
seed = None

for base in bases:
    for dist in distributions:
        # Open a fresh log for this run
        log_path = os.path.join(RESULTS_DIR, f"training_base{base}_dist{dist}.log")
        log_file = open(log_path, "w", buffering=1)
        sys.stdout = log_file

        print(f"Training with base={base}, distribution={dist}")

        tokenizer = MathTokenizer(base)
        pad_id = tokenizer.token2id[tokenizer.pad_token]

        def collate_fn(batch):
            src_batch, tgt_batch = zip(*batch)
            src = nn.utils.rnn.pad_sequence(src_batch, padding_value=pad_id, batch_first=True)
            tgt = nn.utils.rnn.pad_sequence(tgt_batch, padding_value=pad_id, batch_first=True)
            return src, tgt

        model = GCDTransformer(tokenizer, d_model=hidden_dimension, nhead=heads,
                               num_layers=layers, max_length=length)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

        counter = 0
        for epoch in range(100):
            model.train()
            total_loss = 0
            total_sequences = 0
            perfect_sequences = 0

            train_dataset = GCDDataset(max_num=max_int, num_samples=sample_size,
                                       seed=seed, base=base, distribution=dist)
            train_dataloader = DataLoader(train_dataset, batch_size=batch, collate_fn=collate_fn)

            for src, tgt in train_dataloader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                optimizer.zero_grad()
                output = model(src, tgt_input)
                loss = criterion(output.reshape(-1, output.size(-1)),
                                 tgt_output.reshape(-1))
                loss.backward()
                optimizer.step()

                preds = output.argmax(-1)
                mask = (tgt_output != pad_id)
                with torch.no_grad():
                    seq_match = (preds == tgt_output) | ~mask
                    perfect_sequences += seq_match.all(dim=1).sum().item()
                    total_sequences += tgt.size(0)

                total_loss += loss.item()

            counter += 1
            avg_loss = total_loss / len(train_dataloader)
            train_acc = perfect_sequences / max(total_sequences, 1)
            print(f"Epoch {epoch+1}, Loss: {avg_loss}, Accuracy: {train_acc}")

            if counter % validate_step == 0:
                validation_dataset = GCDDataset(max_num=max_int, num_samples=sample_size,
                                                seed=seed, base=base, distribution=dist)
                validation_dataloader = DataLoader(validation_dataset, batch_size=batch,
                                                   collate_fn=collate_fn)
                val_acc = compute_accuracy(model, validation_dataloader, device)
                print(f"Validation Accuracy: {val_acc}")

        test_dataset = GCDDataset(max_num=max_int, num_samples=sample_size,
                                 seed=seed, base=base, distribution='uniform')
        test_dataloader = DataLoader(test_dataset, batch_size=batch, collate_fn=collate_fn)
        test_acc = compute_accuracy(model, test_dataloader, device)
        print(f"Test Accuracy: {test_acc}")

        # Close log and restore stdout
        log_file.close()
        sys.stdout = sys.__stdout__

        # Pickle the trained model
        model_path = os.path.join(RESULTS_DIR, f"model_base{base}_dist{dist}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Save metrics to JSON
        metrics = {
            "base": base,
            "distribution": dist,
            "average_loss": avg_loss,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc
        }
        metrics_path = os.path.join(RESULTS_DIR, f"results_base{base}_dist{dist}.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)