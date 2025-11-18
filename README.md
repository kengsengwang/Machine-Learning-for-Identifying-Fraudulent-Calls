# Machine Learning for Identifying Fraudulent Calls

This project builds an end-to-end machine learning pipeline to classify phone calls as **“Scam”** or **“Not Scam”** using historical call records.  

It follows the AI Singapore AIAP requirements:

- All pipeline logic in **Python `.py` files** under `src/`
- A single **shell script** (`run.sh`) to execute the full pipeline
- Clear instructions to reproduce results on the examiner’s machine

---

## 📁 Project Structure

Final repository structure (as required by the examiner):

```text
Machine-Learning-for-Identifying-Fraudulent-Calls/
├── .github/
│   └── workflows/            # (Optional) helper scripts – not needed to run pipeline
├── src/
│   ├── __init__.py           # Makes src a package (for python -m src.xxx)
│   ├── build_features.py     # Step 1 – load raw data and create processed_features.csv
│   ├── train_and_evaluate.py # Step 2 – train models and generate reports
│   └── utils.py              # Shared logging & helper utilities
├── eda.ipynb                 # Exploratory Data Analysis notebook (for reference)
├── run.sh                    # Convenience script to run the full pipeline
├── requirements.txt          # Python dependencies
└── README.md                 # This file
````

> **Note:**
> The dataset and artifacts are **not committed to Git** (to keep the repo small and clean), but the code expects:
>
> * Raw SQLite DB: `data/calls.db`
> * Processed CSV: `data/processed_calls.csv` (created automatically)
> * Model reports: `artifacts/*.txt` (created automatically)

---

## 📦 Dependencies

* Python **3.11** (recommended for AIAP)
* `pip` and a virtual environment (e.g. `venv`, Conda, or Miniforge)

Install Python packages via:

```bash
pip install -r requirements.txt
```

Key libraries:

* `pandas`, `numpy` – data handling
* `scikit-learn` – preprocessing, models, evaluation
* `sqlite3` (standard library) – reading from `calls.db`
* `joblib` (if later used for model persistence)

---

## 🔄 End-to-End Pipeline Overview

The ML workflow is split into two main Python modules:

1. **`src.build_features`**

   * Connects to `data/calls.db`
   * Automatically detects the call table
   * Engineers features:

     * Time-based features: `call_hour`, `call_dayofweek`, `call_is_weekend`
     * Binary target: `scam_label` from `"Scam Call"` column (`Scam` → 1, `Not Scam` → 0)
   * Writes processed dataset to: `data/processed_calls.csv`

2. **`src.train_and_evaluate`**

   * Loads `data/processed_calls.csv`
   * Splits into features `X` and target `y = scam_label`
   * Separates **numeric** and **categorical** columns
   * Builds a preprocessing pipeline:

     * Numeric: `StandardScaler`
     * Categorical: `OneHotEncoder(handle_unknown="ignore", sparse=False)`
   * Trains and evaluates three models:

     * Logistic Regression
     * Random Forest
     * Gradient Boosting
   * Saves per-model reports under `artifacts/` and a summary file

All logging is handled by helpers in `src.utils`.

---

## 🧪 How to Run the Project

### 1️⃣ Prepare environment

```bash
# From repo root
python -m venv venv        # or use conda/miniforge
source venv/bin/activate   # Linux/Mac
# OR
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

### 2️⃣ Place the data

Create a `data/` folder and copy the provided SQLite database into it:

```text
Machine-Learning-for-Identifying-Fraudulent-Calls/
└── data/
    └── calls.db
```

> The table name is detected automatically; no need to hard-code it.

### 3️⃣ (Recommended) Run the entire pipeline with `run.sh`

From the project root:

```bash
bash run.sh
```

This will:

1. **Step 1 – Build processed dataset**

   ```bash
   python -m src.build_features
   ```

   Output: `data/processed_calls.csv`

2. **Step 2 – Train and evaluate models**

   ```bash
   python -m src.train_and_evaluate
   ```

   Outputs under `artifacts/`:

   * `log_reg_report.txt`
   * `random_forest_report.txt`
   * `grad_boost_report.txt`
   * `summary.txt`

### 4️⃣ (Optional) Run steps individually

If the examiner wants to run each step manually:

```bash
# Step 1 only
python -m src.build_features

# Step 2 only
python -m src.train_and_evaluate
```

---

## 📊 Model Performance (Example Run)

On the provided dataset (`12000` calls, 80/20 train–test split), all three models achieve very strong performance.

Example test-set metrics:

| Model          | Accuracy | Precision (scam) | Recall (scam) | F1 (scam) | ROC AUC |
| -------------- | -------- | ---------------- | ------------- | --------- | ------- |
| Logistic Reg.  | 1.0000   | 1.0000           | 1.0000        | 1.0000    | 1.0000  |
| Random Forest  | 1.0000   | 1.0000           | 1.0000        | 1.0000    | 1.0000  |
| Grad. Boosting | 1.0000   | 1.0000           | 1.0000        | 1.0000    | 1.0000  |

Per-model classification reports are stored in the corresponding `*_report.txt` files under `artifacts/`.

> **Note:** Exact numbers may vary slightly if the random seed or train–test split is changed.

---

## 🧪 EDA Notebook (Optional for Examiner)

The notebook `eda.ipynb` documents exploratory steps:

* Inspection of raw columns from `calls.db`
* Class distribution of scam vs non-scam calls
* Exploration of time-of-day patterns
* Correlations between features and target

The EDA notebook is **not required** to run the pipeline but demonstrates reasoning behind the feature engineering design.

---

## 🔮 Possible Future Improvements

* Add **model persistence** (save best model with `joblib` for serving).
* Implement **hyperparameter tuning** (e.g. `GridSearchCV` or `RandomizedSearchCV`).
* Add **threshold analysis / calibration** for better control of precision–recall trade-offs.
* Extend features (e.g. call duration buckets, number prefixes, country/region info).
* Add unit tests around feature engineering and data loading.

---

## 👤 Author

**Name:** Keng Seng Wang
**Email:** [kengsengwang@outlook.com](mailto:kengsengwang@outlook.com)
**GitHub:** [kengsengwang](https://github.com/kengsengwang)

This project is prepared as part of the **AI Singapore – AI Apprenticeship Programme (AIAP) Technical Assessment**.




