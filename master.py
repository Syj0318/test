#master.py

import pandas as pd
import pymysql
import json
import os
import subprocess

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

# Function to partition data for workers
def partition_data(ticker_data, num_workers):
    partition_size = len(ticker_data) // num_workers
    partitions = []
    for i in range(num_workers):
        start_index = i * partition_size
        end_index = (i + 1) * partition_size if i != num_workers - 1 else len(ticker_data)
        partitions.append(ticker_data[start_index:end_index])
    return partitions

if __name__ == "__main__":
    selected_ticker = "AAPL"  # Change to desired ticker symbol
    ticker_data = fetch_data(selected_ticker)

    # Number of worker instances
    num_workers = 2  # Set to 2 for this example

    # Worker instance IP addresses
    worker_ips = [
        "44.223.93.71",
        "44.203.232.126"
    ]

    # Partition data for each worker
    partitions = partition_data(ticker_data, num_workers)

    # List to hold worker processes
    worker_processes = []

    # Start worker instances
    for worker_id in range(num_workers):
        # Save partition to a temporary file
        partition_file = f'worker_data_{worker_id}.json'
        partitions[worker_id].to_json(partition_file, orient='records')

        # Check if the partition file was created
        if os.path.exists(partition_file):
            print(f"Partition file {partition_file} created successfully.")
        else:
            print(f"Error: Partition file {partition_file} not found.")
            continue  # Skip starting the worker if the file doesn't exist
        
        # Copy the partition file to the worker instance using SCP
        scp_command = f"scp -i COMP4651.pem {partition_file} ubuntu@{worker_ips[worker_id]}:~/test/"
        print(f"Copying {partition_file} to worker {worker_id} with command: {scp_command}")
        
        # Execute the SCP command
        scp_process = subprocess.run(scp_command, shell=True)
        if scp_process.returncode != 0:
            print(f"Error: Failed to copy {partition_file} to worker {worker_id}.")
            continue  # Skip starting the worker if the copy fails

        # Use SSH to start the worker process on the remote instance
        ssh_command = f"ssh -i COMP4651.pem ubuntu@{worker_ips[worker_id]} 'python3 ~/test/worker.py {worker_id} ~/test/{partition_file}'"
        print(f"Starting worker {worker_id} with command: {ssh_command}")

        # Start the worker process and store the process object
        process = subprocess.Popen(ssh_command, shell=True)
        worker_processes.append(process)

    # Wait for all worker processes to finish
    for process in worker_processes:
        process.wait()

    print("All workers have completed. Check output files for results.")