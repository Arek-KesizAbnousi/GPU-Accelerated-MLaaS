# utils/data_ingestion.py

import os
import cudf
import pandas as pd
from tensorflow.keras.datasets import mnist

def ingest_data():
    os.makedirs('data/processed', exist_ok=True)  # Ensure the processed folder exists

    # Load MNIST data
    (x_train, y_train), _ = mnist.load_data()

    # Flatten images
    x_train = x_train.reshape(-1, 28*28)

    # Convert to pandas DataFrame
    df = pd.DataFrame(x_train)
    df['label'] = y_train

    # Convert to cuDF DataFrame for GPU processing
    gdf = cudf.from_pandas(df)

    # Save ingested data
    gdf.to_csv('data/processed/mnist_ingested.csv', index=False)

if __name__ == '__main__':
    ingest_data()
