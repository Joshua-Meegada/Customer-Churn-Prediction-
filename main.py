import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# Auto-detect file in the current directory
current_dir = os.getcwd()
default_file_name = "D:\Coustomer churn prediction\WA_Fn-UseC_-Telco-Customer-Churn.csv"
file_path = os.path.join(current_dir, default_file_name)

# If file not found, ask user to specify the path
if not os.path.exists(file_path):
    print(f"File not found in {current_dir}. Please provide the correct path.")
    file_path = input("D:\Coustomer churn prediction\WA_Fn-UseC_-Telco-Customer-Churn.csv ").strip()
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please check the file path.")

data = pd.read_csv(file_path)

# Drop customerID column as it's not useful for prediction
data.drop(columns=['customerID'], inplace=True)

# Convert TotalCharges to numeric, coerce errors to NaN
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

# Fill missing values with median
data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)

# Encode categorical features
label_encoders = {}
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('Churn')  # Keep target variable separate
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Encode target variable
data['Churn'] = data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Define features and target
X = data.drop(columns=['Churn'])
y = data['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:\n', classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Save model and scaler for future use
joblib.dump(model, 'churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')