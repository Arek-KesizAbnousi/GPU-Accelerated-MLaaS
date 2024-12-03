# GPU-Accelerated MLaaS Platform with NVIDIA RAPIDS

## Introduction

This project showcases the development of a **Machine Learning as a Service (MLaaS) platform** that integrates **data ingestion**, **data labeling**, **visualization**, and **dashboards**. Leveraging **NVIDIA RAPIDS**, **CUDA**, and **PyTorch**, the platform accelerates data processing and model training, demonstrating the effectiveness of GPU acceleration in machine learning workflows.

## Objectives

- **Develop an MLaaS platform** with essential components.
- **Accelerate data processing and model training** using GPU computing.
- **Improve data labeling efficiency** through an interactive tool.
- **Provide real-time visualization** and **dashboard monitoring** for model performance.

## Dataset

- **MNIST Dataset**
  - **Description:** A collection of 70,000 handwritten digit images (60,000 training and 10,000 testing).
  - **Source:** [Yann LeCun's website](http://yann.lecun.com/exdb/mnist/)
  - **Usage:** Utilized for image classification tasks to demonstrate data ingestion, preprocessing, labeling, and model training.

## Architecture

- **Data Ingestion and Preprocessing:**
  - Utilized **Python (Pandas)** and **NVIDIA RAPIDS cuDF** for efficient data handling and preprocessing.

- **Data Labeling Tool:**
  - Built with **Flask** and **Dash** to create an interactive web interface for labeling data samples.

- **Visualization and Dashboards:**
  - Developed using **Dash** and **Plotly** for real-time monitoring of model training metrics.

- **Model Training:**
  - Implemented a **PyTorch** CNN model with GPU support to achieve high accuracy on the MNIST dataset.

## Installation and Setup

### Prerequisites

- **Hardware:**
  - NVIDIA GPU with CUDA capability.
  
- **Software:**
  - Python 3.8+
  - CUDA Toolkit and NVIDIA Drivers.
  - (Optional) Elasticsearch if you plan to implement data indexing.
  
- **Python Packages:**
  - Listed in `requirements.txt`.

### Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/YourUsername/gpu-accelerated-mlaas.git
   cd gpu-accelerated-mlaas
   
2. **Set Up Python Environment**
   ```bash
   conda create -n gpu-mlaas python=3.8
   conda activate gpu-mlaas
   pip install -r requirements.txt
3. **Download and Prepare the Dataset**
   ```bash 
   python utils/data_ingestion.py
   python utils/data_preprocessing.py
  
4. **Run the Data Labeling Tool**
   ```bash
   python utils/data_labeling.py
  - Access the Tool:
    - Open your browser and navigate to `http://localhost:5000.`
  
5. **Train the Model**
   ```bash
   python train.py

6. **Run the Performance Dashboard**
   ```bash
   python dashboards/performance_dashboard.py
  - Access the Dashboard:
      - Open your browser and navigate to `http://localhost:8050.`
 
   
