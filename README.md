# Machine Learning for Identifying Fraudulent Calls

This project aims to identify fraudulent calls using machine learning techniques. It includes data preprocessing, feature engineering, model training, and evaluation.

---

## 📁 Project Structure

```
Machine-Learning-for-Identifying-Fraudulent-Calls/
├── data/
│   └── cleaned_calls.csv
├── scripts/
│   └── class_distribution_smote.py
├── outputs/
│   └── class_distribution_smote.png
├── requirements.txt
└── README.md
```

---

## 📝 Requirements

To install the required packages, run:

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

1. **Clone the repository:**

```bash
git clone https://github.com/kengsengwang/Machine-Learning-for-Identifying-Fraudulent-Calls.git
cd Machine-Learning-for-Identifying-Fraudulent-Calls
```

2. **Install the required packages:**

```bash
pip install -r requirements.txt
```

3. **Run the SMOTE script:**

```bash
python scripts/class_distribution_smote.py
```

4. **Check the outputs:**

The class distribution plots will be saved in the `outputs/` directory as `class_distribution_smote.png`.

---

## 📊 Results

The SMOTE script will generate two bar plots:

- **Class Distribution Before SMOTE**
- **Class Distribution After SMOTE**

These plots will help visualize the impact of SMOTE on the class distribution.

---

## 📌 Future Improvements

- Implement more advanced oversampling techniques.
- Add hyperparameter tuning for the model.
- Include more robust data preprocessing.

---

## 📬 Contact

For questions or support, please contact **Keng Seng Wang** at [kengsengwang@outlook.com](mailto:kengsengwang@outlook.com).
