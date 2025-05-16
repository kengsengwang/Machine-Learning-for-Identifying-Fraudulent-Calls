import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_and_prepare_data(file_path):
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


def main():
    # Example usage:
    file_path = 'cleaned_calls.csv'  # Adjust the file path accordingly
    X, y, preprocessor = load_and_prepare_data(file_path)
    
    # Optionally, you can apply the preprocessor here or save it as a pipeline for later use
    # For example, transforming the features
    X_transformed = preprocessor.fit_transform(X)
    print("Data transformation complete, shape of transformed features:", X_transformed.shape)

if __name__ == "__main__":
    main()
