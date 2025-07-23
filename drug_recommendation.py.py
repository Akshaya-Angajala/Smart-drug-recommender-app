
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# --------------------------------------
# 1. Create a small fake dataset
# --------------------------------------
data = {
    'age': [25, 65, 45, 55, 30, 70, 60, 40, 50, 35],
    'weight': [60, 80, 70, 85, 65, 90, 75, 68, 77, 66],
    'blood_pressure': [0, 1, 1, 1, 0, 1, 1, 0, 1, 0],  # 0=normal, 1=high
    'diabetes': [0, 1, 0, 1, 0, 1, 1, 0, 1, 0],        # 0=no, 1=yes
    'allergy': [0, 0, 1, 0, 1, 0, 0, 1, 0, 1],         # 0=no, 1=yes
    'current_meds': [0, 1, 0, 1, 0, 1, 0, 0, 1, 0],    # 0=no, 1=yes
    'drug': ['Paracetamol', 'Insulin', 'Amoxicillin', 'Insulin', 'Paracetamol',
             'Insulin', 'Insulin', 'Amoxicillin', 'Insulin', 'Paracetamol'],
    'dosage': [500, 10, 500, 12, 650, 14, 10, 500, 13, 650]  # mg or units
}

df = pd.DataFrame(data)

# --------------------------------------
# 2. Train the Decision Tree
# --------------------------------------
features = ['age', 'weight', 'blood_pressure', 'diabetes', 'allergy', 'current_meds']
X = df[features]
y = df[['drug', 'dosage']]

from sklearn.preprocessing import LabelEncoder
le_drug = LabelEncoder()
y['drug_encoded'] = le_drug.fit_transform(y['drug'])

clf_drug = DecisionTreeClassifier(max_depth=3, random_state=42)
clf_drug.fit(X, y['drug_encoded'])

clf_dosage = DecisionTreeClassifier(max_depth=3, random_state=42)
clf_dosage.fit(X, y['dosage'])

# --------------------------------------
# 3. Take new patient input & predict
# --------------------------------------
print("\n--- Patient Input ---")
age = int(input("Age: "))
weight = int(input("Weight (kg): "))
bp = input("Blood Pressure (normal/high): ").lower()
bp_val = 0 if bp == 'normal' else 1
diabetes = input("Diabetes (yes/no): ").lower()
diabetes_val = 1 if diabetes == 'yes' else 0
allergy = input("Any drug allergy (yes/no): ").lower()
allergy_val = 1 if allergy == 'yes' else 0
current_meds = input("Taking other medications (yes/no): ").lower()
current_meds_val = 1 if current_meds == 'yes' else 0

input_features = np.array([[age, weight, bp_val, diabetes_val, allergy_val, current_meds_val]])

drug_encoded = clf_drug.predict(input_features)[0]
predicted_drug = le_drug.inverse_transform([drug_encoded])[0]

predicted_dosage = clf_dosage.predict(input_features)[0]

# --------------------------------------
# 4. Explain WHY
# --------------------------------------
reasons = []
if bp_val == 1:
    reasons.append("high BP")
if diabetes_val == 1:
    reasons.append("diabetes")
if allergy_val == 1:
    reasons.append("has drug allergy")
if current_meds_val == 1:
    reasons.append("currently on other meds")

reason_text = " & ".join(reasons) if reasons else "general mild condition"

# --------------------------------------
# 5. Show output
# --------------------------------------
print("\n--- Recommendation ---")
print(f"Recommended Drug: {predicted_drug}")
print(f"Dosage: {predicted_dosage} mg/units")
if allergy_val == 1 and predicted_drug == "Amoxicillin":
    print("⚠ WARNING: Patient has allergy risk with Amoxicillin!")
print(f"Reason: Recommended due to {reason_text}.")
