import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # =====================
    # Feature Engineering
    # =====================

    # Convert timestamp to datetime and extract features
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df['Month'] = df['Timestamp'].dt.month
    df['Weekend'] = (df['DayOfWeek'] >= 5).astype(int)
    df['BusinessHours'] = ((df['Hour'] >= 9) & (df['Hour'] <= 17)).astype(int)

    # Handle negative call durations (assuming they should be absolute values)
    df['Call Duration'] = df['Call Duration'].abs()

    # Create binary flag for financial loss
    df['HasFinancialLoss'] = (df['Financial Loss'] > 0).astype(int)

    # Create interaction features
    df['SuspiciousInternational'] = (df['Flagged by Carrier'].isin(['Suspicious', 'Very Suspicious'])) & \
                                   (df['Is International'] == 'Yes')
    df['VoipLongCall'] = (df['Call Type'] == 'Voip') & (df['Call Duration'] > 300)

    # Standardize WhatsApp naming
    df['Call Type'] = df['Call Type'].replace({'Whats App': 'WhatsApp'})

    # Handle missing values in Financial Loss
    df['Financial Loss'] = df['Financial Loss'].fillna(0)

    # Cap extreme financial loss values
    financial_loss_cap = df['Financial Loss'].quantile(0.99)
    df['Financial Loss'] = df['Financial Loss'].clip(upper=financial_loss_cap)

    # =====================
    # Data Preparation
    # =====================

    # Define features and target
    X = df.drop(['ID', 'Timestamp', 'Scam Call'], axis=1)
    y = df['Scam Call'].map({'Scam': 1, 'Not Scam': 0})

    # Define numeric and categorical features
    numeric_features = ['Call Duration', 'Call Frequency', 'Financial Loss', 
                       'Previous Contact Count', 'Hour', 'DayOfWeek', 'Month']
    categorical_features = ['Flagged by Carrier', 'Is International', 'Call Type', 
                           'Country Prefix', 'Device Battery', 'Weekend', 
                           'BusinessHours', 'HasFinancialLoss', 'SuspiciousInternational', 
                           'VoipLongCall']

    # Preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    return X, y, preprocessor


def train_model(X, y, preprocessor):
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Define Random Forest model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])

    # Model training
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Model evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%\n")

    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Scam', 'Scam'], 
                yticklabels=['Not Scam', 'Scam'])
    plt.title('Random Forest Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    # Feature Importance
    feature_importance = model.named_steps['classifier'].feature_importances_
    feature_names = numeric_features + list(model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features))
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
    plt.title('Random Forest - Feature Importance')
    plt.tight_layout()
    plt.show()


def main():
    # Example usage:
    file_path = '/mnt/data/cleaned_calls.csv'  # Adjust the file path accordingly
    X, y, preprocessor = load_and_preprocess_data(file_path)
    
    # Train model
    train_model(X, y, preprocessor)


if __name__ == "__main__":
    main()
