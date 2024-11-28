import pandas as pd
import numpy as np
import pymysql
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import json
import tensorflow as tf

# Dataset class definition
class TimeSeriesDataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def get_data(self):
        return self.X, self.y

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
def evaluate_and_save_results(model, X_test, y_test):
    mse = evaluate_model(model, X_test, y_test)
    print(f"Evaluation -> MSE: {mse:.4f}")

    # Save evaluation results to a local file
    results = {'MSE': mse}
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f)
    print(f'Evaluation results saved locally as evaluation_results.json')

if __name__ == "__main__":
    selected_ticker = "AAPL"  # Change to desired ticker symbol
    ticker_data = fetch_data(selected_ticker)

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
    evaluate_and_save_results(model, X_test, y_test)
