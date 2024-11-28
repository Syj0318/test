import pandas as pd
import numpy as np
import json
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import sys

# Dataset class definition
class TimeSeriesDataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def get_data(self):
        return self.X, self.y

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

# Function to evaluate and save results locally
def evaluate_and_save_results(model, X_test, y_test, worker_id):
    mse = evaluate_model(model, X_test, y_test)
    print(f"Worker {worker_id} Evaluation -> MSE: {mse:.4f}")

    # Save evaluation results to a local file
    results = {'worker_id': worker_id, 'MSE': mse}
    with open(f'evaluation_results_worker_{worker_id}.json', 'w') as f:
        json.dump(results, f)
    print(f'Worker {worker_id} evaluation results saved locally as evaluation_results_worker_{worker_id}.json')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 worker.py {worker_id} {partition_file}")
        sys.exit(1)

    worker_id = int(sys.argv[1])
    partition_file = sys.argv[2]

    # Load partitioned data
    ticker_data = pd.read_json(partition_file)

    # Data preprocessing
    X, y = preprocess_data(ticker_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reshape data for LSTM
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

    # Create and train the model
    model = create_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Evaluate models after training
    evaluate_and_save_results(model, X_test, y_test, worker_id)