# Scam Call Detection Project

## Overview

This project aims to develop a machine learning model to classify phone calls as either "scam" or "not scam" based on historical call data. The goal is to enhance the security of phone communications by identifying potential scam calls and automatically triggering alerts for users.

## Dataset

The dataset used for this project is a collection of historical call records, which includes information such as:

- **Call Duration**: Duration of the call.
- **Call Frequency**: Number of calls made by the user.
- **Flagged by Carrier**: Whether the call was flagged by the carrier as suspicious.
- **Is International**: Whether the call was international.
- **Scam Call**: The target variable indicating whether the call is a scam or not.

The dataset is loaded and preprocessed in the project to handle missing values, scale numerical features, and split the data into training and testing sets.

## Files

- **`eda.ipynb`**: This notebook contains the exploratory data analysis (EDA) for understanding the dataset, visualizing features, and investigating correlations.
- **`cleaned_calls.csv`**: This is the cleaned dataset used in the analysis and model training.
- **`requirements.txt`**: The file contains the dependencies required to run the project.

## Requirements

To run the project, you'll need the following dependencies. You can install them using `pip`:

```bash
pip install -r requirements.txt

Installation

    Clone the Repository

    Clone this repository:

cd "C:\Users\DELL\Documents\Machine Learning for Identifying Fraudulent Calls"
git clone https://github.com/kengsengwang/projectname.git
git clone https://github.com/kengsengwang/fraud-detection.git

Install the Required Packages

Install the dependencies listed in requirements.txt:

    pip install -r requirements.txt

Usage
1. Setup Jupyter Notebook

Ensure that you have Jupyter installed:

pip install jupyter

To start Jupyter Notebook:

jupyter notebook

Open the eda.ipynb file in Jupyter and run the cells to explore the dataset, perform EDA, and visualize the data.
2. Model Training

In the eda.ipynb file, we have implemented various machine learning models including Logistic Regression, Random Forest, and Gradient Boosting to predict whether a call is a scam or not.

You can choose to run any of the following steps:

    Preprocessing: Handle missing data, scale numerical features, and encode categorical features.

    Model Training: Train models on the preprocessed dataset.

    Evaluation: Evaluate the model using metrics such as accuracy, precision, recall, and confusion matrix.

3. Model Evaluation

The model's performance is evaluated using the following metrics:

    Accuracy: Percentage of correctly classified calls.

    Confusion Matrix: Visual representation of true positives, false positives, true negatives, and false negatives.

To display the confusion matrix, use the following code:

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Scam', 'Scam'], 
            yticklabels=['Not Scam', 'Scam'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

Future Work

If time, cost, and energy permit, the following areas could be improved:

    Model Optimization: Experiment with more advanced models like XGBoost or Neural Networks for higher accuracy.

    Hyperparameter Tuning: Use techniques like GridSearchCV or RandomizedSearchCV to fine-tune model parameters.

    Feature Engineering: Incorporate additional features or external datasets to improve the model's performance.

    Deployment: Deploy the model as a real-time API service for scam detection.

Contributions

Feel free to fork the repository and submit pull requests with improvements or fixes. All contributions are welcome!
License

This project is licensed under the MIT License - see the LICENSE file for details.
Contact

For any inquiries or feedback, please reach out to wangkengseng@gmail.com or visit my https://www.linkedin.com/in/wang-keng-seng-b5168221/
