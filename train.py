import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

# 1. Load the dataset
# Ensure the CSV file is in the same directory or provide the full path
df = pd.read_csv('your_loan_data.csv') 

# 2. Preprocessing: Categorical Encoding
# Convert categorical features into dummy/indicator variables
df_ml = pd.get_dummies(df, columns=['sex', 'education', 'marriage'], drop_first=True)

# 3. Separate Features (X) and Target (y)
# Drop unique identifiers and the target label from the feature set
X = df_ml.drop(['label', 'id'], axis=1, errors='ignore')
y = df_ml['label']

# 4. Data Splitting (Train-Test Split)
# Using stratify=y to maintain the class distribution in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Model Initialization and Training
# Setting class_weight='balanced' to handle potential class imbalance in loan data
model = RandomForestClassifier(
    n_estimators=100, 
    random_state=42, 
    class_weight='balanced'
)
model.fit(X_train, y_train)

# 6. Model Evaluation with Custom Threshold
# Instead of simple .predict(), we use .predict_proba() to adjust sensitivity
y_probs = model.predict_proba(X_test)[:, 1]

# Adjust threshold from 0.5 to 0.3 to catch more potential defaults (Recall optimization)
custom_threshold = 0.3
y_pred_custom = (y_probs >= custom_threshold).astype(int)

print(f"--- Classification Report (Threshold: {custom_threshold}) ---")
print(classification_report(y_test, y_pred_custom))

# 7. Model Persistence (Export)
# Save the trained model for future deployment (e.g., Vertex AI Model Registry)
joblib.dump(model, 'loan_model.joblib')
print("Model saved as loan_model.joblib")

# 8. Visualization: Confusion Matrix
# Plotting the confusion matrix to evaluate classification performance
cm = confusion_matrix(y_test, y_pred_custom)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Loan Default Prediction')
plt.show()