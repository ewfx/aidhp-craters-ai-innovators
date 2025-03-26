import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTE  # For handling class imbalance
import joblib  # For saving and loading the model

# Step 1: Load Data
data = pd.read_csv("resources/customer_data_generated.csv")

# Step 2: Data Preprocessing
# Handling missing values
data.fillna(data.median(numeric_only=True), inplace=True)

# Encoding categorical variables
label_encoders = {}
categorical_columns = ['marital_status', 'employment_status', 'education_level', 'home_ownership', 'loan_purpose', 'state']
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Store encoders for later use

# Selecting features and target variable
features = [
    'age', 'income', 'account_balance', 'existing_loans', 
    'marital_status', 'employment_status', 'education_level', 
    'debt_to_income_ratio', 'home_ownership', 'employment_duration', 
    'loan_purpose', 'state', 'dependents', 'annual_expenses'
]

target = 'applied_for_home_loan'

X = data[features]
y = data[target]

# Step 3: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Handle Class Imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Step 5: Feature Scaling (Only for numerical features)
scaler = StandardScaler()
numerical_features = [
    'age', 'income', 'account_balance', 'existing_loans', 
    'debt_to_income_ratio', 'employment_duration', 'dependents', 'annual_expenses'
]
X_train_resampled[numerical_features] = scaler.fit_transform(X_train_resampled[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])


# Step 6: Hyperparameter Tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_grid=param_grid,
    cv=3,
    scoring='f1',
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train_resampled, y_train_resampled)
best_model = grid_search.best_estimator_

print("Best Parameters:", grid_search.best_params_)

# Save the trained model, scaler, and label encoders to the models folder
joblib.dump(best_model, "models/home_loan_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(label_encoders, "models/label_encoders.pkl")

print("Best model, scaler, and label encoders have been saved.")

# Step 7.1: Identify the most important features
feature_importances = best_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(feature_importance_df)

# Optionally, save the feature importance to a CSV file
feature_importance_df.to_csv("feature_importances.csv", index=False)

# Step 8: Evaluate Model
y_pred = best_model.predict(X_test)
y_probs = best_model.predict_proba(X_test)[:, 1]  # Probabilities for class 1

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_probs))

# Step 9: Threshold Tuning
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = 2 * (precision * recall) / (precision + recall)
best_threshold = thresholds[np.argmax(f1_scores)]

print("Best Threshold for F1-Score:", best_threshold)

# Predict using the new threshold
y_pred_new = (y_probs >= best_threshold).astype(int)
print("Classification Report with Tuned Threshold:\n", classification_report(y_test, y_pred_new))

# Save metrics to a text file
metrics_output_file = "models/model_metrics.txt"

with open(metrics_output_file, "w") as f:
    f.write("Model Evaluation Metrics\n")
    f.write("========================\n")
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    f.write("\nClassification Report:\n")
    f.write(classification_report(y_test, y_pred))
    f.write(f"\nROC-AUC Score: {roc_auc_score(y_test, y_probs):.4f}\n")
    f.write("\nThreshold Tuning:\n")
    f.write(f"Best Threshold for F1-Score: {best_threshold:.4f}\n")
    f.write("\nClassification Report with Tuned Threshold:\n")
    f.write(classification_report(y_test, y_pred_new))

print(f"Model evaluation metrics have been saved to {metrics_output_file}.")