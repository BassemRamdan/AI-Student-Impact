import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score

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

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("ord", OrdinalEncoder(categories=[ordinal_specs[c] for c in ordinal_cols]), ordinal_cols),
        ("nom", OneHotEncoder(handle_unknown="ignore", drop="first"), nominal_cols),
        ("bool", "passthrough", boolean_cols),
    ],
    remainder="drop",
)

X_reg = df[numerical_cols + ordinal_cols + nominal_cols + boolean_cols]
y_burnout = df["Burnout_Risk_Level"]

X_proc = preprocessor.fit_transform(X_reg)
label_encoder = LabelEncoder()
y_burnout_enc = label_encoder.fit_transform(y_burnout)

# Try different models
print("Trying deeper XGBoost...")
sample_weights = compute_sample_weight(class_weight='balanced', y=y_burnout_enc)
m = xgb.XGBClassifier(n_estimators=600, max_depth=12, learning_rate=0.1, random_state=42)
m.fit(X_proc, y_burnout_enc, sample_weight=sample_weights)

preds = m.predict(X_proc)
print("Accuracy (Deeper):", accuracy_score(y_burnout_enc, preds))

from sklearn.ensemble import RandomForestClassifier
print("Trying RandomForest...")
rf = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, class_weight='balanced')
rf.fit(X_proc, y_burnout_enc)
preds_rf = rf.predict(X_proc)
print("Accuracy (RF):", accuracy_score(y_burnout_enc, preds_rf))

