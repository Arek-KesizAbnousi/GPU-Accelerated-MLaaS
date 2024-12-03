# utils/data_preprocessing.py

# Uncomment the appropriate library: cudf for GPU acceleration or pandas for CPU-based processing
# import cudf  # NVIDIA RAPIDS library for GPU-accelerated DataFrames
import pandas as pd  # Standard library for CPU-based DataFrames

def preprocess_data():
    # Load ingested data
    # Uncomment one of the following lines depending on the library:
    # gdf = cudf.read_csv('data/processed/mnist_ingested.csv')  # Use cudf for GPU-accelerated I/O
    df = pd.read_csv('data/processed/mnist_ingested.csv')  # Use pandas for CPU-based I/O

    # Normalize pixel values
    pixel_columns = [str(i) for i in range(28 * 28)]
    # Uncomment one of the following lines depending on the library:
    # gdf[pixel_columns] = gdf[pixel_columns] / 255.0  # Use cudf for GPU-accelerated processing
    df[pixel_columns] = df[pixel_columns] / 255.0  # Use pandas for CPU-based processing

    # Save the preprocessed data
    # Uncomment one of the following lines depending on the library:
    # gdf.to_csv('data/processed/mnist_preprocessed.csv', index=False)  # Use cudf for GPU-accelerated I/O
    df.to_csv('data/processed/mnist_preprocessed.csv', index=False)  # Use pandas for CPU-based I/O

if __name__ == '__main__':
    preprocess_data()
