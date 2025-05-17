import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import os

# File paths
data_file = 'data/cleaned_calls.csv'
output_image_path = 'outputs/class_distribution_smote.png'

# Create outputs directory if not exists
os.makedirs('outputs', exist_ok=True)

# Load the cleaned_calls dataset
print("Loading data...")
data = pd.read_csv(data_file)

# Check the class distribution before applying SMOTE
class_distribution_before = data['y'].value_counts()

# Split the data into features and target
X = data.drop(columns=['y'])
y = data['y']

# Apply SMOTE to balance the classes
print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Check the class distribution after applying SMOTE
class_distribution_after = pd.Series(y_resampled).value_counts()

# Plot the class distribution before and after applying SMOTE
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.barplot(x=class_distribution_before.index, y=class_distribution_before.values, palette='viridis')
plt.title('Class Distribution Before SMOTE')
plt.xlabel('Class')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.barplot(x=class_distribution_after.index, y=class_distribution_after.values, palette='viridis')
plt.title('Class Distribution After SMOTE')
plt.xlabel('Class')
plt.ylabel('Frequency')

plt.tight_layout()

# Save the plot as an image
plt.savefig(output_image_path)
plt.show()

print(f"Class distribution plots saved to {output_image_path}")
