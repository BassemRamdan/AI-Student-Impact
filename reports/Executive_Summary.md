<div align="center" style="background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%); padding: 40px; border-radius: 20px; border: 1px solid rgba(0, 210, 255, 0.3); font-family: sans-serif; color: white;">
  
  <h1 style="color: #00d2ff; font-size: 3rem; text-shadow: 0 0 15px #00d2ff; margin-bottom: 10px;">Executive Project Report</h1>
  <p style="color: #39ff14; font-size: 1.2rem; margin-top: 0;">AI Student Impact Analysis</p>

</div>

<br>

### 1. Introduction
This report summarizes the comprehensive end-to-end data science lifecycle executed for the **AI Student Impact Analysis** project. The primary objective was to investigate the influence of Generative AI tools on students' academic performance (GPA), skill retention, and psychological well-being (Burnout Risk). 

We transformed a raw dataset into a fully functional, highly interactive **Cyberpunk-themed Dashboard** that empowers educators and researchers to make data-driven decisions.

<hr style="opacity: 0.2;">

### 2. Exploratory Data Analysis (EDA)
In the initial phase, we conducted thorough exploratory analysis to understand the distributions and correlations within our dataset.

- **Univariate Analysis:** We utilized Plotly histograms to analyze the distribution of GPA, traditional study hours, and weekly GenAI hours.
- **Bivariate Analysis:** We mapped the relationship between GenAI usage and its impact on Skill Retention, discovering key thresholds where over-reliance on AI correlates with diminished foundational skills.
- **Visual Design:** All EDA outputs were embedded into the Streamlit dashboard using Custom HTML Glassmorphism tables and neon-styled Plotly charts.

<hr style="opacity: 0.2;">

### 3. Feature Engineering
To maximize the predictive power of our Machine Learning models, we engineered several critical features:
- `GPA_Improvement`: The delta between Post-Semester and Pre-Semester GPA.
- `AI_Usage_Ratio`: The proportion of GenAI hours compared to total study hours.
- `Dependency_per_Hour`: A metric measuring the psychological reliance on AI normalized by usage volume.
- `Burnout_Index`: A composite index formulated from 'Anxiety Levels' and 'Perceived AI Dependency'.

<hr style="opacity: 0.2;">

### 4. Predictive Modeling (Machine Learning)
We developed three distinct **XGBoost** models to capture different facets of student outcomes:

#### A. Post-Semester GPA Prediction (Regression)
- **Algorithm:** `XGBRegressor`
- **Purpose:** Predicts a student's final GPA based on their study habits and AI usage.
- **Performance:** Evaluated using RMSE and R-squared, showing high accuracy in mapping study efficiency to grades.

#### B. Skill Retention Score (Regression)
- **Algorithm:** `XGBRegressor`
- **Purpose:** Measures the negative/positive impact of AI on cognitive retention.

#### C. Burnout Risk Classification
- **Algorithm:** `XGBClassifier`
- **Purpose:** Categorizes students into `Low`, `Medium`, or `High` burnout risk tiers.
- **Dashboard Integration:** Displays predictions via glowing, color-coded Glassmorphism cards (Neon Green for Safe, Pink for Critical).

<hr style="opacity: 0.2;">

### 5. Unsupervised Clustering (Student Personas)
We employed **K-Means Clustering** to segment the student population into unique behavioral personas. The clusters were analyzed and officially named based on their distinct traits:

1. **The AI Pioneers:** High GenAI hours, advanced prompt skills, high GPA.
2. **The Traditional Scholars:** Low AI usage, high traditional study hours.
3. **The Balanced Achievers:** Optimal mix of AI and traditional studying.
4. **The Stressed Adopters:** High AI dependency coupled with high anxiety.
5. **The Casual Users:** Low overall academic engagement.
6. **The Tech Enthusiasts:** High tool diversity, moderate study hours.
7. **The Cautious Learners:** Slow to adopt AI, high skill retention.
8. **The Hyper-dependent:** Severe reliance on AI, resulting in critical burnout risks.

<div align="center">
  <img src="https://img.shields.io/badge/Status-Project%20Complete-39FF14?style=for-the-badge" alt="Status Complete"/>
</div>
