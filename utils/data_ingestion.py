# utils/data_ingestion.py

# Uncomment the appropriate library: cudf for GPU acceleration or pandas for CPU-based processing
# import cudf  # NVIDIA RAPIDS library for GPU-accelerated DataFrames
import pandas as pd  # Standard library for CPU-based DataFrames
from tensorflow.keras.datasets import mnist

def ingest_data():
    # Load MNIST data
    (x_train, y_train), _ = mnist.load_data()

    # Flatten images
    x_train = x_train.reshape(-1, 28 * 28)

    # Convert to pandas DataFrame
    df = pd.DataFrame(x_train)
    df['label'] = y_train

    # If using cudf, convert pandas DataFrame to cudf DataFrame
    # gdf = cudf.from_pandas(df)

    # Save ingested data
    # Uncomment one of the following lines depending on the library:
    # gdf.to_csv('data/processed/mnist_ingested.csv', index=False)  # Use cudf for GPU-accelerated I/O
    df.to_csv('data/processed/mnist_ingested.csv', index=False)  # Use pandas for CPU-based I/O

if __name__ == '__main__':
    ingest_data()
