import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # for importing modules
from exploration import blockage_data_cleaning
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from . import ml_model # lstm machine learning model class defined in this module


def build_model_inputs(blockage_matrix, input_minutes, predict_minutes, step_minutes):
    """
    Converts blockage matrix to ML model inputs and labels.

    Inputs:
    - blockage_matrix (pd.DataFrame): Matrix with blockage durations per interval.
    - input_minutes (int): Number of past minutes used as model input.
    - predict_minutes (int): Number of future minutes to predict.
    - step_minutes (int): Length of each interval (e.g. 2 minutes).

    Returns:
    - features: (N, input_steps, 1) NumPy array of binary input sequences
    - labels: (N, output_steps) NumPy array of binary output labels
    - meta: List of tuples (crossing_id, start_index) for traceability
    """
    matrix = (blockage_matrix.values > 0).astype(int) # binarize
    num_crossings, total_steps = matrix.shape

    input_steps = input_minutes // step_minutes
    output_steps = predict_minutes // step_minutes
    features, labels, meta = [], [], []

    # go through each crossing
    for crossing_idx in range(num_crossings):
        row = matrix[crossing_idx]

        # sliding window across time to generate our training samples
        for t in range(input_steps, total_steps - output_steps + 1):
            
            # X is window of input steps and Y is the next output steps
            features.append(row[t - input_steps:t].reshape(-1, 1))
            labels.append(row[t:t + output_steps]) # target

            meta.append((blockage_matrix.index[crossing_idx], t)) # crossing and time it's from

    return np.array(features), np.array(labels), meta


def train_one_epoch(model, dataloader, criterion, optimizer):
    """
    Runs one epoch of training.

    Inputs:
    - model (nn.Module): a PyTorch model
    - dataloader (DataLoader): DataLoader for training data
    - criterion: Loss function
    - optimizer: Optimizer object

    Returns:
    - avg_loss (float): the average training loss for the epoch
    """
    model.train()
    total_loss = 0

    # go through each batch 
    for input_batch, target_batch in dataloader:
        optimizer.zero_grad() # clear prev grads
        preds = model(input_batch)
        loss = criterion(preds, target_batch)
        loss.backward() # compute gradients
        optimizer.step() # update weights
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def get_accuracy(preds, targets, threshold):
    """
    Computes accuracy of binary predictions.

    Inputs:
    - preds (Tensor): Raw model outputs
    - targets (Tensor): Ground truth binary labels
    - threshold (float): Sigmoid threshold for classification

    Returns:
    - Float accuracy (0 to 1)
    """
    pred_labels = (preds > threshold).float()
    accuracy = (pred_labels == targets).float().sum() / targets.numel()
    return accuracy


def evaluate(model, dataloader, criterion):
    """
    Evaluates model on a validation set.

    Inputs:
    - model (nn.Module): The PyTorch model
    - dataloader (DataLoader): Validation data
    - criterion: Loss function

    Returns a tuple of (average validation loss, accuracy)
    """
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []

    # go through the batches in the dataloader
    with torch.no_grad():
        for input_batch, target_batch in dataloader:
            # forward pass -- predict future blockages
            preds = model(input_batch)
            loss = criterion(preds, target_batch)
            total_loss += loss.item()

            # convert to probabilites
            all_preds.append(torch.sigmoid(preds))
            all_targets.append(target_batch)
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    return total_loss / len(dataloader), get_accuracy(preds, targets, threshold=0.5)


def train_model(model, train_loader, val_loader, epochs, lr):
    """
    Trains model over multiple epochs and prints progress.

    Inputs:
    - model (nn.Module): Model to train
    - train_loader (DataLoader): Training data
    - val_loader (DataLoader): Validation data
    - epochs (int): Number of training epochs
    - lr (float): Learning rate
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")


def predict_for_crossing(model, blockage_matrix, crossing_id, step_minutes, predict_minutes, meta):
    """
    Runs prediction on a specific crossing using the trained model and
    also prints predictions vs. reality for one sample.

    Inputs:
    - model (nn.Module): Trained model
    - blockage_matrix (pd.DataFrame): Matrix used for prediction
    - crossing_id (str): ID to analyze
    - step_minutes (int): Resolution of each interval
    - predict_minutes (int): Horizon to predict
    - meta (List[Tuple]): Meta info with (crossing_id, start_idx)

    Returns none.
    """
    if crossing_id not in blockage_matrix.index:
        raise ValueError(f"Crossing {crossing_id} not found in matrix.")

    output_steps = predict_minutes // step_minutes
    time_labels = blockage_matrix.columns

    with torch.no_grad():
        for i, (cid, idx) in enumerate(meta):
            if cid == crossing_id:
                x = blockage_matrix.loc[cid].values[idx - 15:idx].reshape(1, -1, 1)
                x_tensor = torch.tensor(x, dtype=torch.float32)
                y_true = blockage_matrix.loc[cid].values[idx:idx + output_steps] > 0
                y_pred = (torch.sigmoid(model(x_tensor)) > 0.5).int().squeeze().tolist()
                y_true = y_true.astype(int).tolist()
                for j in range(output_steps):
                    print(f"{time_labels[idx + j]} → Pred: {y_pred[j]} | True: {y_true[j]}")
                break


# ---------- RUN TRAINING AND PREDICTION ----------
if __name__ == "__main__":
    ### TODO: re-organize this and streamline running process into a function

    # may add some other constants here later
    EPOCHS = 10 # try with 30 later
    LEARNING_RATE = 0.01 # mess around with this
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    input_csv = os.path.join(BASE_DIR, "../clean_data/merged_blockage_data.csv")
    cleaned_csv = os.path.join(BASE_DIR, "../clean_data/blockage_data_with_duration.csv")
    output_dir = os.path.join(BASE_DIR, "../clean_data/month_matrices")
    os.makedirs(output_dir, exist_ok=True)

    # load and preprocess raw blockage data
    blockage_data_cleaned = blockage_data_cleaning.preprocess_blockage_data(input_csv, cleaned_csv)

    dec_matrix = blockage_data_cleaning.create_monthly_blockage_matrix(
        blockage_data_cleaned, output_dir, "06:00", "20:00", "2024-12-01", "2024-12-31")
    print("Created Dec 2024 matrix")
    print(f"This is Dec 2024's first five rows {dec_matrix.head()}")
    print(f"This is its shape: {dec_matrix.shape}")

    jan_matrix = blockage_data_cleaning.create_monthly_blockage_matrix(
        blockage_data_cleaned, output_dir, "06:00", "20:00", "2025-01-01", "2025-01-31")
    print("Created Jan 2025 matrix")

    # build train and validation inputs from each month's data
    X_train, y_train, _ = build_model_inputs(dec_matrix, 30, 20, 2)
    X_val, y_val, meta_val = build_model_inputs(jan_matrix, 30, 20, 2)
    print(f"Train shape: {X_train.shape}, {y_train.shape}")
    print(f"Val shape:   {X_val.shape}, {y_val.shape}")

    # create PyTorch data loaders
    train_loader = DataLoader(TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    ), batch_size=64, shuffle=True)

    val_loader = DataLoader(TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    ), batch_size=64)

    # initialize and train LSTM model
    model = ml_model.LSTMBlockagePredictor(hidden_dim=64)
    train_model(model, train_loader, val_loader, EPOCHS, LEARNING_RATE) 

    # save trained model
    model_path = os.path.join(BASE_DIR, "blockage_lstm.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    # predict blockages for a specific crossing
    crossing_id = "288224V"
    print(f"\nPrediction for crossing {crossing_id} on Jan 1st:")
    predict_for_crossing(model, jan_matrix, crossing_id, step_minutes=2, predict_minutes=20, meta=meta_val)
