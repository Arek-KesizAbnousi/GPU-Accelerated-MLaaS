# utils/data_preprocessing.py

import cudf

def preprocess_data():
    gdf = cudf.read_csv('data/processed/mnist_ingested.csv')
    # Normalize pixel values
    pixel_columns = [str(i) for i in range(28*28)]
    gdf[pixel_columns] = gdf[pixel_columns] / 255.0
    # Save preprocessed data
    gdf.to_csv('data/processed/mnist_preprocessed.csv', index=False)

if __name__ == '__main__':
    preprocess_data()
