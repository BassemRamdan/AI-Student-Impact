import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_sample_weight

# Data loading
df = pd.read_csv("data/ai_student_impact_dataset.csv")

numerical_cols = [
    "Pre_Semester_GPA", "Weekly_GenAI_Hours", "Tool_Diversity",
    "Traditional_Study_Hours", "Perceived_AI_Dependency", "Anxiety_Level_During_Exams"
]

ordinal_specs = {
    "Year_of_Study": ["Freshman", "Sophomore", "Junior", "Senior", "Graduate"],
    "Prompt_Engineering_Skill": ["Beginner", "Intermediate", "Advanced"],
}
nominal_cols = ["Major_Category", "Primary_Use_Case", "Institutional_Policy"]
ordinal_cols = list(ordinal_specs.keys())
boolean_cols = ["Paid_Subscription"]

model_numeric_cols = numerical_cols.copy()

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), model_numeric_cols),
        ("ord", OrdinalEncoder(categories=[ordinal_specs[c] for c in ordinal_cols]), ordinal_cols),
        ("nom", OneHotEncoder(handle_unknown="ignore", drop="first"), nominal_cols),
        ("bool", "passthrough", boolean_cols),
    ],
    remainder="drop",
)

X_reg = df[model_numeric_cols + ordinal_cols + nominal_cols + boolean_cols]
y_gpa = df["Post_Semester_GPA"]
y_skill = df["Skill_Retention_Score"]
y_burnout = df["Burnout_Risk_Level"]

X_proc = preprocessor.fit_transform(X_reg)

# Train GPA model
print("Training GPA Model...")
gpa_model = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1, verbosity=0)
gpa_model.fit(X_proc, y_gpa)

# Train Skill model
print("Training Skill Retention Model...")
skill_model = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1, verbosity=0)
skill_model.fit(X_proc, y_skill)

# Train Burnout classifier
print("Training Burnout Risk Model...")
label_encoder = LabelEncoder()
y_burnout_enc = label_encoder.fit_transform(y_burnout)

burnout_model = xgb.XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1, verbosity=0, eval_metric="mlogloss")
sample_weights = compute_sample_weight(class_weight='balanced', y=y_burnout_enc)
burnout_model.fit(X_proc, y_burnout_enc, sample_weight=sample_weights)

# Save everything
print("Saving models...")
with open("models/preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)

with open("models/gpa_model.pkl", "wb") as f:
    pickle.dump(gpa_model, f)

with open("models/skill_model.pkl", "wb") as f:
    pickle.dump(skill_model, f)

with open("models/burnout_model.pkl", "wb") as f:
    pickle.dump(burnout_model, f)

with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("Done!")
