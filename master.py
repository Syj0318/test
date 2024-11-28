import os
import sys
import subprocess

# Install pymysql and boto3 if necessary
subprocess.check_call([sys.executable, "-m", "pip", "install", "pymysql"])

import pandas as pd
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pymysql
import time

# Dataset class definition
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# LSTM model definition
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Last time step output
        return out

# Function to set up distributed training
def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# Function to fetch data from the database
def fetch_data(selected_ticker):
    connection = pymysql.connect(
        host='project-db.cbegtabjn5eh.us-east-1.rds.amazonaws.com',
        user='admin',
        password='comp4651',
        database='stock_price',
        cursorclass=pymysql.cursors.DictCursor
    )

    fetch_query = f"""
        SELECT
            adj_close_price AS `Adj Close`,
            close_price AS `Close`,
            high_price AS `High`,
            low_price AS `Low`,
            open_price AS `Open`,
            volume AS `Volume`,
            price_date AS `price_date`
        FROM daily_price
        WHERE ticker_id = '{selected_ticker}'
        ORDER BY price_date ASC
    """

    try:
        with connection.cursor() as cursor:
            cursor.execute(fetch_query)
            rows = cursor.fetchall()
            ticker_data = pd.DataFrame(rows)

        print("Data fetched successfully.")
        return ticker_data

    except Exception as e:
        print("An error occurred:", e)
    finally:
        connection.close()

# Function for data preprocessing
def preprocess_data(ticker_data):
    relevant_data = ticker_data[['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']]
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(relevant_data)

    sequence_length = 10

    def create_sequences(data, sequence_length):
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length, :])
            y.append(data[i + sequence_length, 0])  # Predict Adj Close
        return np.array(X), np.array(y)

    X, y = create_sequences(normalized_data, sequence_length)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Function to save the model locally
def save_model(model, model_name):
    # Save the model locally
    torch.save(model.state_dict(), model_name)
    print(f'Model {model_name} saved locally successfully.')

# DDP training function
def train(rank, world_size, X_train, y_train, X_test, y_test):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    input_size = X_train.shape[2]
    hidden_size = 32
    output_size = 1
    learning_rate = 0.01
    batch_size = 32
    epochs = 10

    train_dataset = TimeSeriesDataset(X_train, y_train)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

    model = LSTMModel(input_size, hidden_size, output_size).to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Start measuring training time
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_sampler.set_epoch(epoch)  # Ensure proper shuffling
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(rank), y_batch.to(rank)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Rank {rank}, Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

    # End measuring training time
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Rank {rank} Training Time: {training_time:.2f} seconds")

    # Save model locally
    if rank == 0:  # Only the first process uploads
        save_model(model, f'model_rank_{rank}.pth')

    cleanup()

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        X_test = X_test.to(rank)
        y_test = y_test.to(rank)
        predictions = model(X_test)
        mse = nn.MSELoss()(predictions.squeeze(), y_test)
        return mse.item()

# Function to evaluate and save results locally
def evaluate_and_save_results(rank, model, X_test, y_test):
    mse = evaluate_model(model, X_test, y_test)
    print(f"Rank {rank} Evaluation -> MSE: {mse:.4f}")

    # Save evaluation results to a local file
    results = {'MSE': mse}
    with open(f'evaluation_results_rank_{rank}.json', 'w') as f:
        json.dump(results, f)
    print(f'Evaluation results saved locally as evaluation_results_rank_{rank}.json')

if __name__ == "__main__":
    selected_ticker = "AAPL"  # Change to desired ticker symbol
    ticker_data = fetch_data(selected_ticker)

    # Data preprocessing
    X, y = preprocess_data(ticker_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    world_size = torch.cuda.device_count()  # Number of GPUs available

    # Start DDP training
    torch.multiprocessing.spawn(
        train,
        args=(world_size, X_train, y_train, X_test, y_test),
        nprocs=world_size
    )

    # Evaluate models after training
    for rank in range(world_size):
        model = LSTMModel(X_train.shape[2], 32, 1).to(rank)  # Load the model
        model.load_state_dict(torch.load(f'model_rank_{rank}.pth'))  # Load model state
        evaluate_and_save_results(rank, model, X_test, y_test)  # Evaluate and save results
