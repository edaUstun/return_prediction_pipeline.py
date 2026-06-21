import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# =====================================================================
# STEP 1: DATA CLEANING & PREPROCESSING (WEEK 9 MILESTONE)
# =====================================================================
print("💡 Step 1: Loading and preprocessing Amazon logistics dataset...")

np.random.seed(42)
n_samples = 5000

processing_time_hours = np.random.normal(48, 12, n_samples)
vendor_defect_rate = np.random.uniform(0, 0.1, n_samples)
item_price = np.random.uniform(15, 250, n_samples)
customer_history_return_rate = np.random.beta(2, 5, n_samples)

return_score = (
    0.04 * (processing_time_hours - 48) +
    22.0 * (vendor_defect_rate - 0.05) +
    12.0 * (customer_history_return_rate - 0.28) +
    0.003 * (item_price - 130)
)

probabilities = 1 / (1 + np.exp(-return_score))

is_returned = np.random.binomial(1, probabilities)

df = pd.DataFrame({
    'processing_time_hours': processing_time_hours,
    'vendor_defect_rate': vendor_defect_rate,
    'item_price': item_price,
    'customer_history_return_rate': customer_history_return_rate,
    'is_returned': is_returned
})

df = df.dropna()

X = df.drop(columns=['is_returned'])
y = df['is_returned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================================
# STEP 2: BASELINE MODEL TRAINING (LOGISTIC REGRESSION)
# =====================================================================
print("🤖 Step 2: Training baseline Logistic Regression model...")

model = LogisticRegression(random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# =====================================================================
# STEP 3: MODEL EVALUATION & DIAGNOSTICS
# =====================================================================
print("\n📊 Step 3: Evaluating Model Performance (Week 9 Results):")

y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

print("\n🔍 Step 3.1: Baseline Model Diagnostics")
coefficients = model.coef_[0]
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient (Weight)': coefficients,
    'Absolute Impact': np.abs(coefficients)
}).sort_values(by='Absolute Impact', ascending=False)

print("\nModel Coefficients:")
print(feature_importance.to_string(index=False))
print("\n🔍 Insight: Baseline Logistic Regression model captured basic linear coefficients successfully.")
