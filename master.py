import os
import sys
import subprocess
import json

# Install required packages using sys.executable
subprocess.check_call([sys.executable, "-m", "pip", "install", "pymysql"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])  # CPU 버전의 PyTorch 설치
subprocess.check_call([sys.executable, "-m", "pip", "install", "torchvision"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "torchaudio"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "boto3"])

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pymysql
import boto3
import time

s3 = boto3.client('s3',
                  aws_access_key_id='YOUR_ACCESS_KEY',
                  aws_secret_access_key='YOUR_SECRET_KEY',
                  region_name='YOUR_REGION')

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

# Function to save the model to S3
def save_model_to_s3(model, bucket_name, model_name):
    # Save the model locally
    torch.save(model.state_dict(), model_name)

    # Upload to S3
    s3.upload_file(model_name, bucket_name, model_name)
    print(f'Model {model_name} uploaded to S3 successfully.')

# Function to train the model
def train(X_train, y_train):
    input_size = X_train.shape[2]
    hidden_size = 32
    output_size = 1
    learning_rate = 0.01
    batch_size = 32
    epochs = 10

    train_dataset = TimeSeriesDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = LSTMModel(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Start measuring training time
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

    # End measuring training time
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training Time: {training_time:.2f} seconds")

    # Save model to S3
    save_model_to_s3(model, 'your-bucket-name', 'model.pth')

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        mse = nn.MSELoss()(predictions.squeeze(), y_test)
        return mse.item()

# Main execution
if __name__ == "__main__":
    selected_ticker = "AAPL"  # Change to desired ticker symbol
    ticker_data = fetch_data(selected_ticker)

    # Data preprocessing
    X, y = preprocess_data(ticker_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    train(X_train, y_train)

    # Evaluate the model
    model = LSTMModel(X_train.shape[2], 32, 1)  # Load the model
    model.load_state_dict(torch.load('model.pth'))  # Load model state
    mse = evaluate_model(model, X_test, y_test)  # Evaluate
    print(f"Evaluation MSE: {mse:.4f}")
