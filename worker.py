# worker.py

import pandas as pd
import numpy as np
import json
import tensorflow as tf
import sys
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import subprocess

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
    return np.array(X), np.array(y)

# Function to create and compile the LSTM model
def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    mse = model.evaluate(X_test, y_test, verbose=0)
    return mse

if __name__ == "__main__":
    # Get the worker ID and partition file path from command line arguments
    if len(sys.argv) != 3:
        print("Usage: python worker.py <worker_id> <partition_file>")
        sys.exit(1)

    worker_id = int(sys.argv[1])
    partition_file = sys.argv[2]

    # Load the worker data
    ticker_data = pd.read_json(partition_file)

    # Data preprocessing
    X, y = preprocess_data(ticker_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reshape data for LSTM
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

    # Use MirroredStrategy for distributed training
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        # Create and train the model
        model = create_model((X_train.shape[1], X_train.shape[2]))

        # Measure training time
        start_time = time.time()
        model.fit(X_train, y_train, epochs=10, batch_size=32)
        end_time = time.time()

        # Calculate training time
        training_time = end_time - start_time

    # Evaluate the model
    mse = evaluate_model(model, X_test, y_test)

    # Save results to a JSON file
    results = {'Worker ID': worker_id, 'MSE': mse, 'Training Time': training_time}
    result_file = f'results_worker_{worker_id}.json'
    with open(result_file, 'w') as f:
        json.dump(results, f)

    print(f'Worker {worker_id} finished processing and saved results to {result_file}.')

    # Send results back to the master instance using SCP
    scp_command = f"scp -i ~/.ssh/mykey {result_file} ubuntu@44.223.78.166:~/test/"
    print(f"Sending results to master with command: {scp_command}")
    
    # Execute the SCP command
    scp_process = subprocess.run(scp_command, shell=True)
    if scp_process.returncode != 0:
        print(f"Error: Failed to send {result_file} to the master instance.")
    else:
        print(f"Successfully sent {result_file} to the master instance.")