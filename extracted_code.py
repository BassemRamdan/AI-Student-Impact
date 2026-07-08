# ---- Core ----
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

# ---- Visualization ----
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---- Statistics ----
from scipy import stats
from scipy.stats import f_oneway, chi2_contingency, ttest_ind, shapiro, normaltest
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ---- Preprocessing / Model selection ----
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ---- Regression models ----
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
                               RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier

# ---- Classification models ----
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# ---- Metrics ----
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, confusion_matrix, classification_report)

# ---- Clustering & dimensionality reduction ----
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, Birch, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# ---- Explainability ----
import shap

# ---- Plot styling ----
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 110
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["font.size"] = 11
sns.set_style("whitegrid")
PALETTE = ["#2E5EAA", "#E8743B", "#3CA370", "#C44E52", "#8172B2", "#937860"]
sns.set_palette(PALETTE)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 160)

print("Environment ready.")

=====

DATA_PATH = Path("data/ai_student_impact_dataset.csv")
df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

=====

df.head()

=====

df.tail()

=====

df.sample(5, random_state=RANDOM_STATE)

=====

df.info()

=====

df.describe(include="number").T

=====

df.describe(include="object").T

=====

mem_mb = df.memory_usage(deep=True).sum() / 1024**2
print(f"Memory usage: {mem_mb:.2f} MB")

dup_count = df.duplicated().sum()
missing_total = df.isna().sum().sum()
print(f"Duplicate rows: {dup_count}")
print(f"Total missing values: {missing_total}")

missing_by_col = df.isna().sum()
missing_by_col[missing_by_col > 0]

=====

identifier_cols = ["Student_ID"]
target_cols = ["Post_Semester_GPA", "Skill_Retention_Score", "Burnout_Risk_Level"]

numerical_cols = [
    "Pre_Semester_GPA", "Weekly_GenAI_Hours", "Tool_Diversity",
    "Traditional_Study_Hours", "Perceived_AI_Dependency", "Anxiety_Level_During_Exams"
]

categorical_cols = [
    "Major_Category", "Year_of_Study", "Primary_Use_Case",
    "Prompt_Engineering_Skill", "Paid_Subscription", "Institutional_Policy"
]

print("Identifiers:", identifier_cols)
print("Numerical features:", numerical_cols)
print("Categorical features:", categorical_cols)
print("Targets:", target_cols)

assert set(identifier_cols + numerical_cols + categorical_cols + target_cols) == set(df.columns)
print("\nColumn accounting verified: every column has exactly one role.")

=====

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()
for i, col in enumerate(numerical_cols):
    sns.histplot(df[col], kde=True, ax=axes[i], color=PALETTE[i % len(PALETTE)], edgecolor="white")
    axes[i].set_title(f"Distribution of {col}", fontsize=12, fontweight="bold")
    axes[i].set_xlabel(col)
plt.tight_layout()
plt.suptitle("Univariate Distributions — Numerical Features", y=1.02, fontsize=15, fontweight="bold")
plt.show()

=====

fig, axes = plt.subplots(2, 3, figsize=(18, 9))
axes = axes.ravel()
for i, col in enumerate(numerical_cols):
    sns.boxplot(y=df[col], ax=axes[i], color=PALETTE[i % len(PALETTE)])
    axes[i].set_title(col, fontsize=11, fontweight="bold")
plt.tight_layout()
plt.suptitle("Boxplots — Outlier Screening", y=1.02, fontsize=15, fontweight="bold")
plt.show()

=====

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.violinplot(y=df["Post_Semester_GPA"], ax=axes[0], color=PALETTE[0])
axes[0].set_title("Post_Semester_GPA — Violin", fontweight="bold")
sns.violinplot(y=df["Skill_Retention_Score"], ax=axes[1], color=PALETTE[2])
axes[1].set_title("Skill_Retention_Score — Violin", fontweight="bold")
plt.tight_layout()
plt.show()

=====

plt.figure(figsize=(10, 6))
sns.kdeplot(df["Pre_Semester_GPA"], label="Pre-Semester GPA", fill=True, alpha=0.4)
sns.kdeplot(df["Post_Semester_GPA"], label="Post-Semester GPA", fill=True, alpha=0.4)
plt.title("Pre vs. Post Semester GPA — KDE Overlay", fontweight="bold")
plt.xlabel("GPA")
plt.legend()
plt.show()

=====

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()
for i, col in enumerate(categorical_cols):
    order = df[col].value_counts().index
    sns.countplot(y=df[col], order=order, ax=axes[i], palette=PALETTE)
    axes[i].set_title(f"{col} — Counts", fontweight="bold")
    axes[i].set_xlabel("Count")
    axes[i].set_ylabel("")
plt.tight_layout()
plt.suptitle("Categorical Feature Distributions", y=1.02, fontsize=15, fontweight="bold")
plt.show()

=====

fig, ax = plt.subplots(figsize=(7, 7))
df["Burnout_Risk_Level"].value_counts().reindex(["Low", "Medium", "High"]).plot.pie(
    autopct="%1.1f%%", colors=[PALETTE[2], PALETTE[1], PALETTE[3]], ax=ax,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5}
)
ax.set_ylabel("")
ax.set_title("Burnout Risk Level — Share of Students", fontweight="bold")
plt.show()

=====

corr_cols = numerical_cols + ["Post_Semester_GPA", "Skill_Retention_Score"]
corr = df[corr_cols].corr()

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("Correlation Heatmap — Numerical Features & Regression Targets", fontweight="bold")
plt.tight_layout()
plt.show()

=====

sample_df = df.sample(1500, random_state=RANDOM_STATE)
sns.pairplot(
    sample_df[["Pre_Semester_GPA", "Weekly_GenAI_Hours", "Traditional_Study_Hours",
               "Post_Semester_GPA", "Burnout_Risk_Level"]],
    hue="Burnout_Risk_Level", palette=PALETTE[:3], diag_kind="kde", height=2.3, plot_kws={"alpha": 0.5, "s": 18}
)
plt.suptitle("Pairwise Relationships by Burnout Risk (n=1,500 sample)", y=1.02, fontweight="bold")
plt.show()

=====

sns.jointplot(data=sample_df, x="Weekly_GenAI_Hours", y="Post_Semester_GPA", kind="reg",
              height=7, color=PALETTE[0], scatter_kws={"alpha": 0.4, "s": 15})
plt.suptitle("GenAI Hours vs. Post-Semester GPA (with regression fit)", y=1.02, fontweight="bold")
plt.show()

=====

plt.figure(figsize=(9, 7))
plt.hexbin(df["Traditional_Study_Hours"], df["Post_Semester_GPA"], gridsize=35, cmap="Blues", mincnt=1)
plt.colorbar(label="Student count")
plt.xlabel("Traditional Study Hours / week")
plt.ylabel("Post-Semester GPA")
plt.title("Density: Traditional Study Hours vs. Post-Semester GPA", fontweight="bold")
plt.tight_layout()
plt.show()

=====

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.stripplot(data=df.sample(3000, random_state=RANDOM_STATE), x="Prompt_Engineering_Skill",
              y="Post_Semester_GPA", order=["Beginner", "Intermediate", "Advanced"],
              ax=axes[0], alpha=0.35, palette=PALETTE)
axes[0].set_title("GPA by Prompting Skill (strip plot)", fontweight="bold")

sns.violinplot(data=df, x="Prompt_Engineering_Skill", y="Skill_Retention_Score",
               order=["Beginner", "Intermediate", "Advanced"], ax=axes[1], palette=PALETTE, inner="quartile")
axes[1].set_title("Skill Retention by Prompting Skill (violin)", fontweight="bold")
plt.tight_layout()
plt.show()

=====

fig = px.scatter(
    df.sample(4000, random_state=RANDOM_STATE),
    x="Weekly_GenAI_Hours", y="Skill_Retention_Score", color="Burnout_Risk_Level",
    size="Tool_Diversity", opacity=0.55, hover_data=["Major_Category", "Primary_Use_Case"],
    color_discrete_sequence=[PALETTE[2], PALETTE[1], PALETTE[3]],
    category_orders={"Burnout_Risk_Level": ["Low", "Medium", "High"]},
    title="GenAI Hours vs. Skill Retention, colored by Burnout Risk (bubble size = Tool Diversity)"
)
fig.update_layout(template="plotly_white", height=550)
fig.show()

=====

policy_summary = (
    df.groupby("Institutional_Policy")[["Post_Semester_GPA", "Skill_Retention_Score"]]
    .mean().reset_index()
)
fig = px.bar(
    policy_summary.melt(id_vars="Institutional_Policy", var_name="Metric", value_name="Average"),
    x="Institutional_Policy", y="Average", color="Metric", barmode="group",
    color_discrete_sequence=[PALETTE[0], PALETTE[2]],
    title="Average Outcomes by Institutional AI Policy"
)
fig.update_layout(template="plotly_white", height=500)
fig.show()

=====

def summarize_numeric_vs_target(feature, target="Post_Semester_GPA", bins=10):
    tmp = df[[feature, target]].copy()
    tmp["bucket"] = pd.qcut(tmp[feature], q=bins, duplicates="drop")
    return tmp.groupby("bucket", observed=True)[target].mean()

for feat in numerical_cols:
    trend = summarize_numeric_vs_target(feat)
    direction = "increases" if trend.iloc[-1] > trend.iloc[0] else "decreases"
    print(f"{feat:32s} -> average Post_Semester_GPA {direction} from {trend.iloc[0]:.3f} to {trend.iloc[-1]:.3f} "
          f"across deciles")

=====

fig, ax = plt.subplots(figsize=(10, 6))
hours_bins = pd.cut(df["Weekly_GenAI_Hours"], bins=[0, 5, 10, 15, 20, 25, 40])
trend = df.groupby(hours_bins, observed=True)["Post_Semester_GPA"].mean()
trend.plot(kind="line", marker="o", ax=ax, color=PALETTE[0], linewidth=2.5, markersize=8)
ax.set_title("Average Post-Semester GPA by Weekly GenAI Hours Bucket", fontweight="bold")
ax.set_xlabel("Weekly GenAI Hours")
ax.set_ylabel("Average Post-Semester GPA")
plt.tight_layout()
plt.show()

=====

print(df["Post_Semester_GPA"].describe())
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.histplot(df["Post_Semester_GPA"], kde=True, ax=axes[0], color=PALETTE[0])
axes[0].set_title("Post_Semester_GPA — Distribution", fontweight="bold")
sns.boxplot(x=df["Major_Category"], y=df["Post_Semester_GPA"], ax=axes[1], palette=PALETTE)
axes[1].set_title("Post_Semester_GPA by Major", fontweight="bold")
axes[1].tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.show()

=====

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.histplot(df["Skill_Retention_Score"], kde=True, ax=axes[0], color=PALETTE[2])
axes[0].set_title("Skill_Retention_Score — Distribution", fontweight="bold")
sns.boxplot(x=df["Prompt_Engineering_Skill"], y=df["Skill_Retention_Score"],
            order=["Beginner", "Intermediate", "Advanced"], ax=axes[1], palette=PALETTE)
axes[1].set_title("Skill Retention by Prompting Skill", fontweight="bold")
plt.tight_layout()
plt.show()

=====

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.boxplot(x=df["Burnout_Risk_Level"], y=df["Perceived_AI_Dependency"],
            order=["Low", "Medium", "High"], ax=axes[0], palette=[PALETTE[2], PALETTE[1], PALETTE[3]])
axes[0].set_title("AI Dependency by Burnout Risk", fontweight="bold")
sns.boxplot(x=df["Burnout_Risk_Level"], y=df["Anxiety_Level_During_Exams"],
            order=["Low", "Medium", "High"], ax=axes[1], palette=[PALETTE[2], PALETTE[1], PALETTE[3]])
axes[1].set_title("Exam Anxiety by Burnout Risk", fontweight="bold")
plt.tight_layout()
plt.show()

=====

eng = df.copy()

# GPA_Improvement: did the student's GPA move up or down this semester?
eng["GPA_Improvement"] = eng["Post_Semester_GPA"] - eng["Pre_Semester_GPA"]

# Study_Efficiency: GPA generated per hour of *traditional* study — a proxy for how much a student
# gets out of the study time that doesn't involve AI.
eng["Study_Efficiency"] = eng["Post_Semester_GPA"] / eng["Traditional_Study_Hours"].replace(0, np.nan)
eng["Study_Efficiency"] = eng["Study_Efficiency"].fillna(eng["Study_Efficiency"].median())

# AI_Usage_Ratio: share of total study-like time (AI + traditional) spent using AI.
eng["Total_Study_Hours"] = eng["Weekly_GenAI_Hours"] + eng["Traditional_Study_Hours"]
eng["AI_Usage_Ratio"] = eng["Weekly_GenAI_Hours"] / eng["Total_Study_Hours"].replace(0, np.nan)
eng["AI_Usage_Ratio"] = eng["AI_Usage_Ratio"].fillna(0)

# AI_vs_Traditional: simple relative-intensity ratio (log-scaled to tame the long right tail).
eng["AI_vs_Traditional"] = np.log1p(eng["Weekly_GenAI_Hours"]) - np.log1p(eng["Traditional_Study_Hours"])

# Dependency_per_Hour: is a student's *reported* dependency high relative to how much they actually use AI?
# A high value flags psychological dependency that outstrips actual usage volume.
eng["Dependency_per_Hour"] = eng["Perceived_AI_Dependency"] / (eng["Weekly_GenAI_Hours"] + 1)

# Prompt_Skill_Score: turn the ordinal skill label into a numeric score for downstream engineered ratios.
skill_map = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}
eng["Prompt_Skill_Numeric"] = eng["Prompt_Engineering_Skill"].map(skill_map)

# Tool_Efficiency: skill retained per tool used — rewards students who get more out of fewer tools.
eng["Tool_Efficiency"] = eng["Skill_Retention_Score"] / eng["Tool_Diversity"]

# Burnout_Index: a composite of the two variables Section 7 showed track burnout most closely.
eng["Burnout_Index"] = (
    0.5 * eng["Perceived_AI_Dependency"] / 10 + 0.5 * eng["Anxiety_Level_During_Exams"] / 10
) * 100

# Academic_Efficiency: GPA improvement per combined hour invested (AI + traditional).
eng["Academic_Efficiency"] = eng["GPA_Improvement"] / eng["Total_Study_Hours"].replace(0, np.nan)
eng["Academic_Efficiency"] = eng["Academic_Efficiency"].fillna(0)

# Skill_per_Study_Hour: skill retention normalized by total study effort — a "learning ROI" metric.
eng["Skill_per_Study_Hour"] = eng["Skill_Retention_Score"] / eng["Total_Study_Hours"].replace(0, np.nan)
eng["Skill_per_Study_Hour"] = eng["Skill_per_Study_Hour"].fillna(eng["Skill_per_Study_Hour"].median())

# Prompt_Weighted_AI_Hours: hours of AI use weighted by how skillfully those hours are likely spent.
eng["Prompt_Weighted_AI_Hours"] = eng["Weekly_GenAI_Hours"] * eng["Prompt_Skill_Numeric"]

engineered_features = [
    "GPA_Improvement", "Study_Efficiency", "AI_Usage_Ratio", "AI_vs_Traditional",
    "Dependency_per_Hour", "Prompt_Skill_Numeric", "Tool_Efficiency", "Burnout_Index",
    "Academic_Efficiency", "Skill_per_Study_Hour", "Prompt_Weighted_AI_Hours", "Total_Study_Hours"
]
eng[engineered_features].describe().T

=====

plt.figure(figsize=(9, 6))
sns.histplot(eng["GPA_Improvement"], kde=True, color=PALETTE[0])
plt.axvline(0, color="black", linestyle="--", linewidth=1)
plt.title("Distribution of GPA_Improvement (Post - Pre)", fontweight="bold")
plt.show()

pct_improved = (eng["GPA_Improvement"] > 0).mean() * 100
print(f"Share of students with a GPA improvement this semester: {pct_improved:.1f}%")

=====

ordinal_specs = {
    "Year_of_Study": ["Freshman", "Sophomore", "Junior", "Senior", "Graduate"],
    "Prompt_Engineering_Skill": ["Beginner", "Intermediate", "Advanced"],
}
nominal_cols = ["Major_Category", "Primary_Use_Case", "Institutional_Policy"]
ordinal_cols = list(ordinal_specs.keys())
boolean_cols = ["Paid_Subscription"]

model_numeric_cols = numerical_cols.copy()  # base numeric features used for modeling

def build_preprocessor(scaler="standard"):
    scaler_obj = {"standard": StandardScaler(), "robust": RobustScaler(), "minmax": MinMaxScaler()}[scaler]
    return ColumnTransformer(
        transformers=[
            ("num", scaler_obj, model_numeric_cols),
            ("ord", OrdinalEncoder(categories=[ordinal_specs[c] for c in ordinal_cols]), ordinal_cols),
            ("nom", OneHotEncoder(handle_unknown="ignore", drop="first"), nominal_cols),
            ("bool", "passthrough", boolean_cols),
        ],
        remainder="drop",
    )

preprocessor = build_preprocessor("standard")
print("Preprocessing pipeline constructed with StandardScaler for numeric features.")

=====

X_reg = df[model_numeric_cols + ordinal_cols + nominal_cols + boolean_cols]
y_gpa = df["Post_Semester_GPA"]
y_skill = df["Skill_Retention_Score"]
y_burnout = df["Burnout_Risk_Level"]

X_train, X_test, y_gpa_train, y_gpa_test = train_test_split(
    X_reg, y_gpa, test_size=0.2, random_state=RANDOM_STATE
)
_, _, y_skill_train, y_skill_test = train_test_split(
    X_reg, y_skill, test_size=0.2, random_state=RANDOM_STATE
)
_, _, y_burn_train, y_burn_test = train_test_split(
    X_reg, y_burnout, test_size=0.2, random_state=RANDOM_STATE, stratify=y_burnout
)

print(f"Train size: {X_train.shape[0]:,} | Test size: {X_test.shape[0]:,}")
print(f"Feature columns going into preprocessing: {X_reg.shape[1]}")

=====

scaler_options = {"StandardScaler": StandardScaler(), "RobustScaler": RobustScaler(), "MinMaxScaler": MinMaxScaler()}
example_col = "Weekly_GenAI_Hours"
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, (name, scaler) in zip(axes, scaler_options.items()):
    scaled = scaler.fit_transform(df[[example_col]])
    sns.histplot(scaled.ravel(), ax=ax, color=PALETTE[0], kde=True)
    ax.set_title(name, fontweight="bold")
plt.suptitle(f"Scaler Comparison on '{example_col}'", y=1.03, fontweight="bold")
plt.tight_layout()
plt.show()

=====

corr_results = []
for col in numerical_cols:
    r, p = stats.pearsonr(df[col], df["Post_Semester_GPA"])
    corr_results.append({"feature": col, "pearson_r": round(r, 3), "p_value": p,
                          "significant_0.05": p < 0.05})
corr_df = pd.DataFrame(corr_results).sort_values("pearson_r", key=abs, ascending=False)
corr_df

=====

anova_results = []
for col in categorical_cols:
    groups = [group["Post_Semester_GPA"].values for _, group in df.groupby(col, observed=True)]
    f_stat, p_val = f_oneway(*groups)
    anova_results.append({"feature": col, "F_statistic": round(f_stat, 2), "p_value": p_val,
                           "significant_0.05": p_val < 0.05})
anova_df = pd.DataFrame(anova_results).sort_values("F_statistic", ascending=False)
anova_df

=====

chi_results = []
for col in categorical_cols:
    ct = pd.crosstab(df[col], df["Burnout_Risk_Level"])
    chi2, p, dof, _ = chi2_contingency(ct)
    chi_results.append({"feature": col, "chi2": round(chi2, 2), "dof": dof, "p_value": p,
                         "significant_0.05": p < 0.05})
chi_df = pd.DataFrame(chi_results).sort_values("chi2", ascending=False)
chi_df

=====

paid = df[df["Paid_Subscription"] == True]
unpaid = df[df["Paid_Subscription"] == False]

for target in ["Post_Semester_GPA", "Skill_Retention_Score"]:
    t_stat, p_val = ttest_ind(paid[target], unpaid[target], equal_var=False)
    print(f"{target:24s} | paid mean={paid[target].mean():.3f}  unpaid mean={unpaid[target].mean():.3f}  "
          f"t={t_stat:.2f}  p={p_val:.4f}  significant={p_val < 0.05}")

=====

normality_results = []
for col in ["Post_Semester_GPA", "Skill_Retention_Score"] + numerical_cols:
    sample = df[col].sample(min(5000, len(df)), random_state=RANDOM_STATE)  # shapiro caps well under 5k for speed
    stat, p = shapiro(sample)
    normality_results.append({"feature": col, "shapiro_stat": round(stat, 4), "p_value": p,
                               "looks_normal (p>0.05)": p > 0.05})
pd.DataFrame(normality_results)

=====

vif_data = df[numerical_cols].copy()
vif_data = (vif_data - vif_data.mean()) / vif_data.std()
vif_data = vif_data.assign(const=1)

vif_results = pd.DataFrame({
    "feature": numerical_cols,
    "VIF": [variance_inflation_factor(vif_data.values, i) for i in range(len(numerical_cols))]
}).sort_values("VIF", ascending=False)
vif_results

=====

outlier_summary = []
for col in numerical_cols:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_summary.append({"feature": col, "lower_bound": round(lower, 2), "upper_bound": round(upper, 2),
                             "n_outliers": n_outliers, "pct_outliers": round(100 * n_outliers / len(df), 2)})
pd.DataFrame(outlier_summary)

=====

preprocessor = build_preprocessor("standard")
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)
print(f"Processed feature matrix shape: {X_train_proc.shape}")

SLOW_MODEL_SAMPLE_SIZE = 6000
rng = np.random.RandomState(RANDOM_STATE)
slow_idx = rng.choice(X_train_proc.shape[0], size=SLOW_MODEL_SAMPLE_SIZE, replace=False)

=====

def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def evaluate_regressor(name, model, X_tr, y_tr, X_te, y_te, cv_X=None, cv_y=None, cv_folds=3):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    mae = mean_absolute_error(y_te, preds)
    mse = mean_squared_error(y_te, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_te, preds)
    adj_r2 = adjusted_r2(r2, X_te.shape[0], X_te.shape[1])
    cv_scores = cross_val_score(model, cv_X if cv_X is not None else X_tr,
                                 cv_y if cv_y is not None else y_tr,
                                 cv=cv_folds, scoring="r2", n_jobs=-1)
    return {"Model": name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2,
            "Adjusted_R2": adj_r2, "CV_R2_mean": cv_scores.mean(), "CV_R2_std": cv_scores.std()}

=====

regressors_gpa = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
    "Lasso": Lasso(alpha=0.001, random_state=RANDOM_STATE),
    "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=-1, random_state=RANDOM_STATE),
    "Extra Trees": ExtraTreesRegressor(n_estimators=200, max_depth=12, n_jobs=-1, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=RANDOM_STATE),
    "XGBoost": xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                 random_state=RANDOM_STATE, n_jobs=-1, verbosity=0),
    "LightGBM": lgb.LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                   random_state=RANDOM_STATE, verbosity=-1),
    "CatBoost": CatBoostRegressor(iterations=300, depth=6, learning_rate=0.05,
                                   random_state=RANDOM_STATE, verbose=False),
}

results_gpa = []
for name, model in regressors_gpa.items():
    res = evaluate_regressor(name, model, X_train_proc, y_gpa_train, X_test_proc, y_gpa_test)
    results_gpa.append(res)
    print(f"{name:20s} done | R2={res['R2']:.4f}")

=====

# Slow models: SVR & KNN fit on a subsample for tractability
slow_regressors_gpa = {
    "SVR": SVR(kernel="rbf", C=1.0, epsilon=0.05),
    "KNN Regressor": KNeighborsRegressor(n_neighbors=15, n_jobs=-1),
}
X_slow, y_slow = X_train_proc[slow_idx], y_gpa_train.values[slow_idx]
for name, model in slow_regressors_gpa.items():
    res = evaluate_regressor(name, model, X_slow, y_slow, X_test_proc, y_gpa_test, cv_folds=3)
    res["Model"] = f"{name} (trained on {SLOW_MODEL_SAMPLE_SIZE:,}-row sample)"
    results_gpa.append(res)
    print(f"{name:20s} done | R2={res['R2']:.4f}")

=====

results_gpa_df = pd.DataFrame(results_gpa).sort_values("R2", ascending=False).reset_index(drop=True)
results_gpa_df.style.background_gradient(subset=["R2", "Adjusted_R2"], cmap="Greens")

=====

plt.figure(figsize=(11, 6))
order = results_gpa_df.sort_values("R2")
plt.barh(order["Model"], order["R2"], color=PALETTE[0])
plt.xlabel("Test R²")
plt.title("Regression Model Comparison — Post_Semester_GPA", fontweight="bold")
plt.tight_layout()
plt.show()

best_gpa_model_name = results_gpa_df.iloc[0]["Model"]
print(f"Best model for Post_Semester_GPA: {best_gpa_model_name}")

=====

_, _, y_skill_train2, y_skill_test2 = train_test_split(
    X_reg, y_skill, test_size=0.2, random_state=RANDOM_STATE
)  # identical split indices as GPA/burnout since X_reg/random_state match

regressors_skill = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
    "Lasso": Lasso(alpha=0.01, random_state=RANDOM_STATE),
    "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=-1, random_state=RANDOM_STATE),
    "Extra Trees": ExtraTreesRegressor(n_estimators=200, max_depth=12, n_jobs=-1, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=RANDOM_STATE),
    "XGBoost": xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                 random_state=RANDOM_STATE, n_jobs=-1, verbosity=0),
    "LightGBM": lgb.LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                   random_state=RANDOM_STATE, verbosity=-1),
    "CatBoost": CatBoostRegressor(iterations=300, depth=6, learning_rate=0.05,
                                   random_state=RANDOM_STATE, verbose=False),
}

results_skill = []
for name, model in regressors_skill.items():
    res = evaluate_regressor(name, model, X_train_proc, y_skill_train2, X_test_proc, y_skill_test2)
    results_skill.append(res)
    print(f"{name:20s} done | R2={res['R2']:.4f}")

y_skill_slow = y_skill_train2.values[slow_idx]
for name, model in slow_regressors_gpa.items():
    model_fresh = SVR(kernel="rbf") if name == "SVR" else KNeighborsRegressor(n_neighbors=15, n_jobs=-1)
    res = evaluate_regressor(name, model_fresh, X_slow, y_skill_slow, X_test_proc, y_skill_test2, cv_folds=3)
    res["Model"] = f"{name} (trained on {SLOW_MODEL_SAMPLE_SIZE:,}-row sample)"
    results_skill.append(res)
    print(f"{name:20s} done | R2={res['R2']:.4f}")

=====

results_skill_df = pd.DataFrame(results_skill).sort_values("R2", ascending=False).reset_index(drop=True)
results_skill_df.style.background_gradient(subset=["R2", "Adjusted_R2"], cmap="Greens")

=====

plt.figure(figsize=(11, 6))
order = results_skill_df.sort_values("R2")
plt.barh(order["Model"], order["R2"], color=PALETTE[2])
plt.xlabel("Test R²")
plt.title("Regression Model Comparison — Skill_Retention_Score", fontweight="bold")
plt.tight_layout()
plt.show()

best_skill_model_name = results_skill_df.iloc[0]["Model"]
print(f"Best model for Skill_Retention_Score: {best_skill_model_name}")

=====

best_gpa_estimator = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                       random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
best_gpa_estimator.fit(X_train_proc, y_gpa_train)

best_skill_estimator = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                         random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
best_skill_estimator.fit(X_train_proc, y_skill_train2)

feature_names_proc = preprocessor.get_feature_names_out()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, model, title, color in zip(
    axes, [best_gpa_estimator, best_skill_estimator],
    ["Feature Importance — GPA (XGBoost)", "Feature Importance — Skill Retention (XGBoost)"],
    [PALETTE[0], PALETTE[2]]
):
    imp = pd.Series(model.feature_importances_, index=feature_names_proc).sort_values(ascending=True).tail(10)
    ax.barh(imp.index, imp.values, color=color)
    ax.set_title(title, fontweight="bold")
plt.tight_layout()
plt.show()

=====

label_encoder = LabelEncoder()
y_burn_train_enc = label_encoder.fit_transform(y_burn_train)
y_burn_test_enc = label_encoder.transform(y_burn_test)
class_names = label_encoder.classes_
print("Classes:", list(class_names))

X_train_c_proc = preprocessor.transform(X_train)   # same split/order as burnout train
X_test_c_proc = preprocessor.transform(X_test)

=====

def evaluate_classifier(name, model, X_tr, y_tr, X_te, y_te, cv_X=None, cv_y=None, cv_folds=3, needs_proba=True):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    acc = accuracy_score(y_te, preds)
    prec = precision_score(y_te, preds, average="weighted", zero_division=0)
    rec = recall_score(y_te, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_te, preds, average="weighted", zero_division=0)
    try:
        proba = model.predict_proba(X_te)
        auc = roc_auc_score(y_te, proba, multi_class="ovr", average="weighted")
    except Exception:
        auc = np.nan
    cv_scores = cross_val_score(model, cv_X if cv_X is not None else X_tr,
                                 cv_y if cv_y is not None else y_tr,
                                 cv=cv_folds, scoring="accuracy", n_jobs=-1)
    return {"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1,
            "ROC_AUC": auc, "CV_Accuracy_mean": cv_scores.mean(), "CV_Accuracy_std": cv_scores.std()}, preds

=====

classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, n_jobs=-1, random_state=RANDOM_STATE),
    "Extra Trees": ExtraTreesClassifier(n_estimators=200, max_depth=12, n_jobs=-1, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=RANDOM_STATE),
    "XGBoost": xgb.XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.05, random_state=RANDOM_STATE,
                                  n_jobs=-1, verbosity=0, eval_metric="mlogloss"),
    "LightGBM": lgb.LGBMClassifier(n_estimators=250, max_depth=5, learning_rate=0.05,
                                    random_state=RANDOM_STATE, verbosity=-1),
    "CatBoost": CatBoostClassifier(iterations=250, depth=6, learning_rate=0.05, random_state=RANDOM_STATE,
                                    verbose=False),
    "Naive Bayes": GaussianNB(),
}

results_burnout = []
predictions_store = {}
for name, model in classifiers.items():
    res, preds = evaluate_classifier(name, model, X_train_c_proc, y_burn_train_enc, X_test_c_proc, y_burn_test_enc)
    results_burnout.append(res)
    predictions_store[name] = preds
    print(f"{name:20s} done | Acc={res['Accuracy']:.4f}  F1={res['F1']:.4f}")

=====

# SVM & KNN — subsample for tractability, same rationale as the regression section
slow_classifiers = {
    "SVM (RBF)": SVC(probability=True, random_state=RANDOM_STATE),
    "KNN Classifier": KNeighborsClassifier(n_neighbors=15, n_jobs=-1),
}
X_slow_c, y_slow_c = X_train_c_proc[slow_idx], y_burn_train_enc[slow_idx]
for name, model in slow_classifiers.items():
    res, preds = evaluate_classifier(name, model, X_slow_c, y_slow_c, X_test_c_proc, y_burn_test_enc, cv_folds=3)
    res["Model"] = f"{name} (trained on {SLOW_MODEL_SAMPLE_SIZE:,}-row sample)"
    results_burnout.append(res)
    predictions_store[res["Model"]] = preds
    print(f"{name:20s} done | Acc={res['Accuracy']:.4f}")

=====

results_burnout_df = pd.DataFrame(results_burnout).sort_values("F1", ascending=False).reset_index(drop=True)
results_burnout_df.style.background_gradient(subset=["Accuracy", "F1", "ROC_AUC"], cmap="Greens")

=====

plt.figure(figsize=(11, 6))
order = results_burnout_df.sort_values("F1")
plt.barh(order["Model"], order["F1"], color=PALETTE[1])
plt.xlabel("Weighted F1 Score")
plt.title("Classifier Comparison — Burnout Risk Level", fontweight="bold")
plt.tight_layout()
plt.show()

best_burnout_model_name = results_burnout_df.iloc[0]["Model"]
print(f"Best classifier for Burnout_Risk_Level: {best_burnout_model_name}")

=====

best_model_key = "XGBoost"  # tree ensembles are the most consistent top performer across our regression sections too
best_clf = classifiers[best_model_key]
best_preds = predictions_store[best_model_key]

cm = confusion_matrix(y_burn_test_enc, best_preds)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix — {best_model_key}", fontweight="bold")
plt.tight_layout()
plt.show()

print(classification_report(y_burn_test_enc, best_preds, target_names=class_names))

=====

best_clf.fit(X_train_c_proc, y_burn_train_enc)
imp = pd.Series(best_clf.feature_importances_, index=feature_names_proc).sort_values(ascending=True).tail(12)
plt.figure(figsize=(9, 7))
plt.barh(imp.index, imp.values, color=PALETTE[3])
plt.title(f"Feature Importance — {best_model_key} (Burnout Risk)", fontweight="bold")
plt.tight_layout()
plt.show()

=====

cluster_features = numerical_cols + ["Post_Semester_GPA", "Skill_Retention_Score"]
cluster_features_full = cluster_features + ["Prompt_Skill_Numeric"]

cluster_source = eng[cluster_features_full].copy()
cluster_scaler = StandardScaler()
X_cluster_full = cluster_scaler.fit_transform(cluster_source)

# Agglomerative/DBSCAN scale poorly to 50k rows (O(n^2) memory/time) -- use a representative random sample.
CLUSTER_SAMPLE_SIZE = 5000
cluster_sample_idx = np.random.RandomState(RANDOM_STATE).choice(len(cluster_source), CLUSTER_SAMPLE_SIZE, replace=False)
X_cluster_sample = X_cluster_full[cluster_sample_idx]

print(f"Full clustering matrix: {X_cluster_full.shape} | Sampled matrix (for O(n^2) methods): {X_cluster_sample.shape}")

=====

inertias = []
k_range = range(2, 9)
for k in k_range:
    km_elbow = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X_cluster_full)
    inertias.append(km_elbow.inertia_)

plt.figure(figsize=(9, 6))
plt.plot(list(k_range), inertias, marker="o", color=PALETTE[0], linewidth=2.5, markersize=8)
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (within-cluster sum of squares)")
plt.title("Elbow Method — KMeans Inertia", fontweight="bold")
plt.tight_layout()
plt.show()

=====

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

metric_rows = []
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X_cluster_full)
    labels_k = km.labels_
    sample_for_sil = np.random.RandomState(RANDOM_STATE).choice(len(X_cluster_full), 5000, replace=False)
    sil = silhouette_score(X_cluster_full[sample_for_sil], labels_k[sample_for_sil])
    db = davies_bouldin_score(X_cluster_full, labels_k)
    ch = calinski_harabasz_score(X_cluster_full, labels_k)
    metric_rows.append({"k": k, "silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch})

metrics_df = pd.DataFrame(metric_rows)
metrics_df

=====

best_k = int(metrics_df.sort_values("silhouette", ascending=False).iloc[0]["k"])
print(f"Selected k = {best_k} (highest average silhouette score)")

=====

kmeans_model = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10).fit(X_cluster_full)
mbkmeans_model = MiniBatchKMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10, batch_size=1024).fit(X_cluster_full)
gmm_model = GaussianMixture(n_components=best_k, random_state=RANDOM_STATE).fit(X_cluster_full)

eng["Cluster_KMeans"] = kmeans_model.labels_
eng["Cluster_MiniBatchKMeans"] = mbkmeans_model.labels_
eng["Cluster_GMM"] = gmm_model.predict(X_cluster_full)

agreement_km_mb = (eng["Cluster_KMeans"] == eng["Cluster_MiniBatchKMeans"]).mean()
print(f"Label agreement, KMeans vs. MiniBatchKMeans: {agreement_km_mb:.1%} (label identities can permute; "
      f"this is an approximate agreement check, not definitive)")

=====

agglo_model = AgglomerativeClustering(n_clusters=best_k).fit(X_cluster_sample)
birch_model = Birch(n_clusters=best_k).fit(X_cluster_sample)

# DBSCAN needs an eps; approximate via a k-distance heuristic rather than guessing blindly.
from sklearn.neighbors import NearestNeighbors
neighbors = NearestNeighbors(n_neighbors=5).fit(X_cluster_sample)
distances, _ = neighbors.kneighbors(X_cluster_sample)
eps_guess = np.percentile(np.sort(distances[:, -1]), 90)
dbscan_model = DBSCAN(eps=eps_guess, min_samples=10).fit(X_cluster_sample)

n_dbscan_clusters = len(set(dbscan_model.labels_)) - (1 if -1 in dbscan_model.labels_ else 0)
n_dbscan_noise = (dbscan_model.labels_ == -1).sum()
print(f"DBSCAN found {n_dbscan_clusters} clusters (eps={eps_guess:.3f}) with {n_dbscan_noise} points labeled noise "
      f"out of {len(X_cluster_sample)}")

=====

pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
proj_2d = pca_2d.fit_transform(X_cluster_full)
eng["PCA1"], eng["PCA2"] = proj_2d[:, 0], proj_2d[:, 1]

plot_sample = eng.sample(6000, random_state=RANDOM_STATE)
fig = px.scatter(
    plot_sample, x="PCA1", y="PCA2", color=plot_sample["Cluster_KMeans"].astype(str),
    opacity=0.55, color_discrete_sequence=PALETTE,
    title=f"KMeans Clusters (k={best_k}) Projected onto First 2 Principal Components"
)
fig.update_layout(template="plotly_white", height=550)
fig.show()

=====

cluster_profile = eng.groupby("Cluster_KMeans")[cluster_features_full].mean().round(2)
cluster_profile["n_students"] = eng["Cluster_KMeans"].value_counts().sort_index()
cluster_profile

=====

burnout_by_cluster = pd.crosstab(eng["Cluster_KMeans"], eng["Burnout_Risk_Level"], normalize="index") * 100
burnout_by_cluster = burnout_by_cluster.reindex(columns=["Low", "Medium", "High"])
burnout_by_cluster.round(1)

=====

pca_full = PCA(random_state=RANDOM_STATE).fit(X_cluster_full)
explained = pca_full.explained_variance_ratio_
cum_explained = np.cumsum(explained)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].bar(range(1, len(explained) + 1), explained, color=PALETTE[0])
axes[0].set_title("Explained Variance per Principal Component", fontweight="bold")
axes[0].set_xlabel("Component")
axes[1].plot(range(1, len(cum_explained) + 1), cum_explained, marker="o", color=PALETTE[1])
axes[1].axhline(0.9, color="black", linestyle="--", linewidth=1)
axes[1].set_title("Cumulative Explained Variance", fontweight="bold")
axes[1].set_xlabel("Number of components")
plt.tight_layout()
plt.show()

n_for_90pct = int(np.argmax(cum_explained >= 0.9) + 1)
print(f"Components needed to explain 90% of variance: {n_for_90pct} out of {X_cluster_full.shape[1]}")

=====

loadings = pd.DataFrame(pca_full.components_[:2].T, index=cluster_features_full, columns=["PC1", "PC2"])
loadings.sort_values("PC1", ascending=False)

=====

TSNE_SAMPLE_SIZE = 3000
tsne_idx = np.random.RandomState(RANDOM_STATE).choice(len(X_cluster_full), TSNE_SAMPLE_SIZE, replace=False)
tsne_model = TSNE(n_components=2, perplexity=35, random_state=RANDOM_STATE, init="pca")
tsne_proj = tsne_model.fit_transform(X_cluster_full[tsne_idx])

tsne_plot_df = eng.iloc[tsne_idx].copy()
tsne_plot_df["TSNE1"], tsne_plot_df["TSNE2"] = tsne_proj[:, 0], tsne_proj[:, 1]

fig = px.scatter(tsne_plot_df, x="TSNE1", y="TSNE2", color="Burnout_Risk_Level",
                  color_discrete_sequence=[PALETTE[2], PALETTE[1], PALETTE[3]],
                  category_orders={"Burnout_Risk_Level": ["Low", "Medium", "High"]}, opacity=0.6,
                  title=f"t-SNE Projection (n={TSNE_SAMPLE_SIZE:,} sample), colored by Burnout Risk")
fig.update_layout(template="plotly_white", height=550)
fig.show()

=====

UMAP_SAMPLE_SIZE = 5000
umap_idx = np.random.RandomState(RANDOM_STATE).choice(len(X_cluster_full), UMAP_SAMPLE_SIZE, replace=False)
umap_model = umap.UMAP(n_components=2, random_state=RANDOM_STATE, n_neighbors=25, min_dist=0.3)
umap_proj = umap_model.fit_transform(X_cluster_full[umap_idx])

umap_plot_df = eng.iloc[umap_idx].copy()
umap_plot_df["UMAP1"], umap_plot_df["UMAP2"] = umap_proj[:, 0], umap_proj[:, 1]

fig = px.scatter(umap_plot_df, x="UMAP1", y="UMAP2", color=umap_plot_df["Cluster_KMeans"].astype(str),
                  color_discrete_sequence=PALETTE, opacity=0.6,
                  title=f"UMAP Projection (n={UMAP_SAMPLE_SIZE:,} sample), colored by KMeans Cluster")
fig.update_layout(template="plotly_white", height=550)
fig.show()

=====

SHAP_SAMPLE_SIZE = 1500
shap_idx = np.random.RandomState(RANDOM_STATE).choice(X_test_proc.shape[0], SHAP_SAMPLE_SIZE, replace=False)
X_test_shap = X_test_proc[shap_idx]
X_test_shap_df = pd.DataFrame(X_test_shap, columns=feature_names_proc)

explainer_gpa = shap.TreeExplainer(best_gpa_estimator)
shap_values_gpa = explainer_gpa.shap_values(X_test_shap_df)
print("SHAP values computed for GPA model, shape:", shap_values_gpa.shape)

=====

shap.summary_plot(shap_values_gpa, X_test_shap_df, show=False, plot_size=(10, 7))
plt.title("SHAP Summary — Post_Semester_GPA Model", fontweight="bold")
plt.tight_layout()
plt.show()

=====

hours_col = "num__Weekly_GenAI_Hours"
if hours_col in X_test_shap_df.columns:
    shap.dependence_plot(hours_col, shap_values_gpa, X_test_shap_df, show=False)
    plt.title("SHAP Dependence — Weekly GenAI Hours vs. GPA Impact", fontweight="bold")
    plt.tight_layout()
    plt.show()

=====

sample_i = 0
shap_explanation = shap.Explanation(
    values=shap_values_gpa[sample_i], base_values=explainer_gpa.expected_value,
    data=X_test_shap_df.iloc[sample_i], feature_names=list(X_test_shap_df.columns)
)
shap.plots.waterfall(shap_explanation, show=False)
plt.title(f"SHAP Waterfall — Individual Student (test row {sample_i})", fontweight="bold")
plt.tight_layout()
plt.show()

=====

explainer_burnout = shap.TreeExplainer(best_clf)
shap_values_burnout = explainer_burnout.shap_values(X_test_shap_df)

# For multiclass, shap_values_burnout has shape (n_samples, n_features, n_classes) in recent SHAP versions
if isinstance(shap_values_burnout, list):
    class_idx = list(class_names).index("High")
    shap.summary_plot(shap_values_burnout[class_idx], X_test_shap_df, show=False, plot_size=(10, 7))
else:
    class_idx = list(class_names).index("High")
    shap.summary_plot(shap_values_burnout[:, :, class_idx], X_test_shap_df, show=False, plot_size=(10, 7))
plt.title("SHAP Summary — Burnout Risk Model (class = High)", fontweight="bold")
plt.tight_layout()
plt.show()