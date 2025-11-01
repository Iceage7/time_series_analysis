# PP Price Direction Prediction

A machine learning pipeline to predict the weekly direction (Up/Down) of Polypropylene (PP) prices.

## 1\. Installation

Clone the repository and install the required dependencies.

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
```


## 2\. Pipeline Usage

The pipeline is run in sequence from the terminal.

### Step 1: Preprocess Data

- **What it does:** Merges, cleans, and time-aligns all raw data sources and creates the target variable.
- **Example:**
  ```bash
  python preprocess.py \
    --pp-file "path/to/raw_pp_prices.csv" \
    --brent-file "path/to/raw_brent_prices.csv" \
    --gas-file "path/to/raw_gas_prices.csv" \
    --output-file "processed_data.csv"
  ```
- **Input:** Raw CSV files for PP, Brent, and Natural Gas.
- **Output:** A single `processed_data.csv` file.

### Step 2: Engineer Features

- **What it does:** Generates lags, rolling statistics, and technical indicators from the processed data.
- **Example:**
  ```bash
  python create_features.py \
    --input-file "processed_data.csv" \
    --output-file "featured_data.csv"
  ```
- **Input:** `processed_data.csv`
- **Output:** `featured_data.csv` (the final model-ready dataset).

### Step 3: Train Model

- **What it does:** Trains the model, evaluates it on a holdout set, and saves the final model artifacts.
- **Example:**
  ```bash
  python train.py \
    --input-file "featured_data.csv" \
    --output-file "training_results.json" \
    --test-size 24 \
    --n-splits 5
  ```
- **Input:** `featured_data.csv`
- **Output:**
  - `training_results.json`: A JSON file with all performance metrics.
  - `final_model.pkl`: The trained model.
  - `scaler.pkl`: The fitted data scaler.
  - `feature_names.pkl`: The list of feature names used by the model.

### Step 4: Visualize Results

- **What it does:** Reads the training results and generates performance plots.
- **Example:**
  ```bash
  python visualize.py \
    --input-file "training_results.json" \
    --output-dir "plots"
  ```
- **Input:** `training_results.json`
- **Output:** PNG/PDF images (e.g., confusion matrix, feature importance) in the `plots/` directory.

### Step 5: Run Inference

- **What it does:** Uses the saved model to make a prediction on the most recent data.
- **Example:**
  ```bash
  python inference.py \
    --pp-file "path/to/raw_pp_prices.csv" \
    --brent-file "path/to/raw_brent_prices.csv" \
    --gas-file "path/to/raw_gas_prices.csv"
  ```
- **Input:** The _updated_ raw data files and the saved model artifacts (`final_model.pkl`, `scaler.pkl`, `feature_names.pkl`).
- **Output:** A prediction printed to the console (e.g., `P(Up)`, `Predicted Class`, `Top Feature Drivers`).
