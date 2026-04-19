# 🏋️ Body Performance Analytics — ML Pipeline

## 🔍 About the Project
An end-to-end Machine Learning project analyzing body performance data
to predict fitness class (A–D) and broad jump distance using various ML models.

## 📊 Dataset
- **Source:** Body Performance Dataset
- **Size:** 13,393 rows × 12 columns
- **Targets:**
  - `class` (A / B / C / D) → Classification
  - `broad jump_cm` → Regression

## 🔬 Pipeline

### 1. Data Preparation & EDA
- Dataset overview and column understanding with medical valid ranges
- Checked data types, missing values (none found), and duplicates (1 removed)
- Data validity checks — removed 494 impossible-value rows
- Univariate analysis (histograms + skewness)
- Bivariate analysis (violin plots, scatter plots, categorical vs target)
- Correlation matrix analysis

### 2. Feature Engineering
| Feature | Description |
|---|---|
| BMI | weight / height² |
| pulse_pressure | systolic − diastolic |
| MAP | (systolic + 2×diastolic) / 3 |
| LBM_kg | weight × (1 − body fat%) |
| BP_category | Normal / Pre_High / High_Stage1 / High_Stage2 |
| fitness_score | Percentile rank across grip + sit-ups + broad jump |

### 3. Outlier Treatment
- IQR capping for most features
- P1–P99 capping for `sit and bend forward_cm`
- Gender-based IQR capping for `body fat_%`
- No rows removed — shape unchanged

### 4. Encoding
- Label Encoding for `gender`, `class`, `BP_category`

### 5. Feature Selection
**Classification features:**
`sit and bend forward_cm`, `sit-ups counts`, `broad jump_cm`, `gripForce`,
`fitness_score`, `body fat_%`, `age`, `weight_kg`, `gender_encoded`

> Selected based on domain knowledge and biological relevance to physical performance.

**Regression features:**
Selected based on correlation threshold (|r| > 0.3) with `broad jump_cm`.
`fitness_score` excluded due to data leakage (derived from target).

### 6. Models

**Classification (target: class A/B/C/D)**
| Model | Tuning Parameter |
|---|---|
| KNN | n_neighbors (k=21, manhattan, distance weights) |
| Decision Tree | max_depth (depth=10, entropy criterion) |
| SVM | kernel (rbf selected over linear and poly) |
| Neural Network | hidden_layer_sizes (100, 50) |

**Regression (target: broad jump_cm)**
| Model | Tuning Parameter |
|---|---|
| Linear Regression | Baseline — no tuning |
| KNN | n_neighbors (k=21, manhattan, distance weights) |
| Decision Tree | max_depth (depth=7) |
| SVM | kernel (rbf selected over linear) |
| Neural Network | hidden_layer_sizes (100, 50) |

## 🏆 Best Results
| Task | Model | Score |
|---|---|---|
| Classification | Neural Network (100,50) | ~72.56% Accuracy |
| Regression | Neural Network (100,50) | ~0.80 R² |

## 🛠️ Tools Used
- Python, Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn, Plotly
- Google Colab / Jupyter Notebook

## 📂 Files
Body-Performance-ML/
│── BodyPerformance_Source_Code.ipynb
│── README.md
