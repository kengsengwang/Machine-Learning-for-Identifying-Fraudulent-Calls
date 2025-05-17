import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Load the data
data_path = './data/cleaned_calls.csv'
df = pd.read_csv(data_path)

# Preprocessing
df['Scam Call'] = df['Scam Call'].map({'Scam': 1, 'Not Scam': 0})
df = pd.get_dummies(df, columns=['Flagged by Carrier', 'Is International', 'Country Prefix', 'Call Type', 'Device Battery'], drop_first=True)

# Splitting the data
X = df.drop(columns=['ID', 'Timestamp', 'Scam Call'])
y = df['Scam Call']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Extract top 3 models
top_3_models = {
    "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

# Train each model
for name, model in top_3_models.items():
    model.fit(X_train, y_train)

# Plotting Feature Importance for the Top 3 Models
plt.figure(figsize=(24, 12))

for i, (name, model) in enumerate(top_3_models.items(), 1):
    # Extract feature importances if available
    if hasattr(model, "feature_importances_"):
        feature_importances = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False).head(15)
        
        # Plot top 15 most important features
        plt.subplot(1, 3, i)
        sns.barplot(x="Importance", y="Feature", data=feature_importances, palette="viridis")
        plt.title(f"Top 15 Feature Importances - {name}")
        plt.xlabel("Importance")
        plt.ylabel("Feature")

plt.tight_layout()
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Load the data
data_path = './data/cleaned_calls.csv'
df = pd.read_csv(data_path)

# Preprocessing
df['Scam Call'] = df['Scam Call'].map({'Scam': 1, 'Not Scam': 0})
df = pd.get_dummies(df, columns=['Flagged by Carrier', 'Is International', 'Country Prefix', 'Call Type', 'Device Battery'], drop_first=True)

# Splitting the data
X = df.drop(columns=['ID', 'Timestamp', 'Scam Call'])
y = df['Scam Call']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Extract top 3 models
top_3_models = {
    "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

# Train each model
for name, model in top_3_models.items():
    model.fit(X_train, y_train)

# Plotting Feature Importance for the Top 3 Models
plt.figure(figsize=(24, 12))

for i, (name, model) in enumerate(top_3_models.items(), 1):
    # Extract feature importances if available
    if hasattr(model, "feature_importances_"):
        feature_importances = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False).head(15)
        
        # Plot top 15 most important features
        plt.subplot(1, 3, i)
        sns.barplot(x="Importance", y="Feature", data=feature_importances, palette="viridis")
        plt.title(f"Top 15 Feature Importances - {name}")
        plt.xlabel("Importance")
        plt.ylabel("Feature")

plt.tight_layout()
plt.show()
