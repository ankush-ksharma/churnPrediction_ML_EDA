# Telco Customer Churn — Exploratory Data Analysis & Modeling

My Approach was to first to EDA, create new features and then check for data coliniriality.
Once i get the final features, use Cross Validation to get best model which will be used for predctions.


## 1. Data Loading & Initial Inspection

- Loaded `data/TelcoCustomerChurn.csv` with pandas.
- Reviewed `df.info()`, `df.describe()`, and categorical summary stats.
- Sped up scikit-learn operations on Intel CPUs via `sklearnex.patch_sklearn()`.

## 2. Data Quality Checks and Misssing Value Analysis and Duplciate Analysis

- **Missing values**: none found — no imputation required.
- **Duplicates**: none found — no removal required.
- **Column typing note**: `SeniorCitizen` is stored as `int` but is conceptually categorical, so it was handled manually in the categorical/numerical split.


## 3. Target Variable Analysis

- Target column: `Churn` (`Yes` / `No`).
- Class distribution is imbalanced — roughly **25–30% churn rate**, meaning accuracy alone is not a sufficient evaluation metric.

## 4. Data Cleaning and nmerical/categorical identifcation

- Dropped `customerID` — an arbitrary unique identifier with no predictive value.
- Converted `TotalCharges` from string to numeric (`pd.to_numeric`, coercing errors), and filled resulting nulls (blank strings) with `0.0`.
- Mapped target `Churn` to binary: `Yes → 1`, `No → 0`.

## 5. Train/Test Split

- Split with `train_test_split`, **70/30**, `random_state=42`, **stratified** on `Churn` to preserve class balance across sets.

## 6. Exploratory Data Analysis (Visual)

Plots generated on the training set only (to avoid leakage): target distribution, tenure density by churn, churn proportion by contract type, monthly charges by churn, churn by internet service tier, and churn by tech support status.

**Key insights:**

- **Target distribution**: moderate class imbalance (~25–30% churn); F1/recall matter more than raw accuracy.
- **Tenure**: churned customers concentrate at low tenure; long-tenure customers churn less. Tenure is a strong predictor.
- **Contract type**: month-to-month has the highest churn (~40%); one-year is much lower; two-year is lowest (~3%). Strongest churn driver.
- **Monthly charges**: churned customers tend to have higher monthly charges (median ~$80).
- **Internet service**: Fiber optic customers churn the most; DSL less; no-internet customers rarely churn.
- **Tech support**: customers without Tech Support churn far more than those with it.

**High-risk profile**: low tenure, month-to-month contract, high monthly charges, fiber optic service, no tech support.
**Low-risk profile**: long tenure, one/two-year contract, lower monthly charges, tech support enabled.

## 7. Feature Engineering (Round 1)

| Feature | Definition | Rationale |
|---|---|---|
| `Service_Count` | Count of add-on services (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`) marked `Yes` | Proxy for "ecosystem stickiness" / switching cost |
| `Average_Monthly_Charges` | `TotalCharges / tenure` (falls back to `MonthlyCharges` when `tenure == 0`) | Proxy for billing consistency; large gap vs. current `MonthlyCharges` can signal an expired promo or recent price hike — a churn trigger |

## 8. Statistical Analysis & Feature Selection

- **Correlation heatmap** on numerical features + Pearson correlation with `Churn`.
- **Chi-square test** on one-hot encoded categorical features to rank association with `Churn`.
- **VIF (multicollinearity) analysis** on numerical features.

**Findings:**

- Churn is mainly driven by **contract type, payment method, internet service, tenure, and monthly charges**.
- **Month-to-month** contracts: ~42.9% churn; **two-year** contracts: ~3.0% churn.
- **Electronic check** payment method has the highest churn rate (~45.7%).
- **Fiber optic** internet and customers lacking **Online Security** / **Tech Support** churn considerably more.
- **Tenure** is the strongest numerical predictor of retention.
- **MonthlyCharges** correlates positively with churn.
- Chi-square confirmed **Contract, PaymentMethod, InternetService, OnlineSecurity, TechSupport, Dependents, SeniorCitizen** as the most influential categorical features.
- Multicollinearity found between `MonthlyCharges`/`Average_Monthly_Charges`, and between `TotalCharges`/`tenure`.

**Decision:** drop `Average_Monthly_Charges` (redundant with `MonthlyCharges`).

## 9. Feature Engineering (Round 2 — Final)

| Feature | Definition |
|---|---|
| `AutoPayment` | 1 if `PaymentMethod` is an automatic bank transfer or credit card, else 0 |
| `FamilyCustomer` | 1 if customer has a `Partner` or `Dependents`, else 0 |
| `CustomerType` | Lifecycle bucket from `tenure`: `New` (≤12 mo), `Regular` (13–36 mo), `Loyal` (>36 mo) |
| `Support_Count` | Count of support-related add-ons (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`) marked `Yes` |

Final numerical columns: `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`, `Service_Count`, `Support_Count`. All remaining columns treated as categorical.

## 10. Preprocessing Pipeline

Built with `ColumnTransformer` (fit only on training data to prevent leakage):

- **Numerical**: `StandardScaler`.
- **Categorical**: `OneHotEncoder(drop='if_binary', handle_unknown='ignore')`.

A dense-output variant of the same transformer (`sparse_output=False`) was used for classifiers that require dense input (e.g., Gradient Boosting family, Naive Bayes, Discriminant Analysis, ensembles/meta-estimators).

## 11. Baseline Model Training — Decision Tree

Three Decision Tree pipelines were trained and compared:

1. **Unpruned Tree** — default `DecisionTreeClassifier(random_state=42)`.
2. **Balanced Tree** — `max_depth=5`, `min_samples_leaf=10`, `class_weight='balanced'`.
3. **Tuned Tree (GridSearchCV)** — grid over `max_depth`, `min_samples_split`, `min_samples_leaf`, 5-fold CV, scored on **F1**.

The best GridSearchCV estimator (`decision_tree_bestGridCV`) was evaluated with accuracy, precision, recall, F1, and a confusion matrix, then **saved to `model/churn_pipeline.pkl`** via `joblib`.

## 12. Bonus: Broad Model Comparison

Trained and tuned (via `GridSearchCV`, 5-fold **stratified** CV, scoring on **F1**) a wide range of additional classifiers, each wrapped in the same preprocessing pipeline:

- Linear: Logistic Regression, Ridge Classifier, SGD Classifier
- Margin-based: Linear SVM, RBF SVM
- Tree ensembles: Random Forest, Extra Trees
- Boosting: Gradient Boosting, HistGradient Boosting, AdaBoost
- Instance-based: K-Nearest Neighbors
- Probabilistic: Gaussian Naive Bayes
- Discriminant analysis: LDA, QDA
- Neural Network (MLP)
- Meta-estimators: Voting Classifier (LR + RF + SVM, soft voting), Stacking Classifier (LR + RF + KNN → LR meta-model)

### Results (sorted by F1 Score, test set)

| Model | Best CV F1 | Precision | Recall | F1 Score | Accuracy |
|---|---|---|---|---|---|
| Random Forest | 0.6241 | 0.5223 | 0.8146 | **0.6365** | 0.7530 |
| SGD Classifier | 0.6301 | 0.5161 | 0.8004 | 0.6275 | 0.7478 |
| Logistic Regression | 0.6277 | 0.5156 | 0.7968 | 0.6261 | 0.7473 |
| Quadratic Discriminant Analysis | 0.6258 | 0.5258 | 0.7629 | 0.6225 | 0.7544 |
| Extra Trees | 0.6191 | 0.5146 | 0.7861 | 0.6220 | 0.7463 |
| Ridge Classifier | 0.6199 | 0.5051 | 0.8021 | 0.6198 | 0.7388 |
| Linear SVM | 0.6228 | 0.5051 | 0.8004 | 0.6193 | 0.7388 |
| Tuned Decision Tree | 0.5720 | 0.6138 | 0.6007 | 0.6072 | 0.7937 |
| Neural Network (MLP) | 0.5874 | 0.5993 | 0.6132 | 0.6062 | 0.7885 |
| Balanced Decision Tree | NaN | 0.4621 | 0.8360 | 0.5952 | 0.6981 |
| AdaBoost | 0.5960 | 0.6347 | 0.5544 | 0.5918 | 0.7970 |
| Stacking Classifier | 0.5979 | 0.6782 | 0.5223 | 0.5901 | **0.8074** |
| Gaussian Naive Bayes | 0.5936 | 0.4477 | 0.8396 | 0.5840 | 0.6824 |
| Linear Discriminant Analysis | 0.5957 | 0.6177 | 0.5472 | 0.5803 | 0.7899 |
| Voting Classifier | 0.5963 | 0.6468 | 0.5223 | 0.5779 | 0.7974 |
| Gradient Boosting | 0.5930 | 0.6597 | 0.5080 | 0.5740 | 0.7998 |
| RBF SVM | 0.5830 | 0.6221 | 0.5312 | 0.5731 | 0.7899 |
| HistGradient Boosting | 0.5857 | 0.6356 | 0.5098 | 0.5658 | 0.7922 |
| K-Nearest Neighbors | 0.5692 | 0.5612 | 0.5312 | 0.5458 | 0.7653 |
| Decision Tree (unpruned) | NaN | 0.4750 | 0.4902 | 0.4825 | 0.7208 |

Confusion matrices were plotted for every model, sorted by F1 Score, for side-by-side visual comparison.

### Key Observations

1. **Best F1 — Random Forest (0.6365)**: best balance of precision/recall, catching >81% of churners.
2. **Best Accuracy — Stacking Classifier (0.8074)**: highest precision but lower recall — trades off missing more churners for fewer false positives.
3. **Linear models are competitive**: Logistic Regression, SGD, Ridge, Linear SVM all cluster around F1 ≈ 0.62, suggesting the problem is largely linearly separable.
4. **Balanced Decision Tree**: highest recall among trees (0.8360) but low precision (0.4621) — aggressive churn flagging.
5. **Unpruned Decision Tree**: weakest model overall (F1 = 0.4825), confirming pruning/balancing was necessary.
6. **Boosting models** favor precision over recall — more conservative, useful when false alarms are costly.

### Recommendation

- **Maximize churn capture** → **Random Forest** (highest F1, strong recall ~81.5%).
- **Minimize false positives / maximize accuracy** → **Stacking Classifier**.
- The originally tuned, interpretable **Decision Tree** (`model/churn_pipeline.pkl`, F1 = 0.6072) remains a solid baseline, though outperformed by several ensembles/linear models.

## 13. Impact of Feature Engineering

Adding `CustomerType` and `Support_Count` produced only minor performance changes versus the earlier feature set. The original raw features (Contract Type, Payment Method, Internet Service, Tech Support, Online Security, tenure, Monthly Charges) already captured most of the churn signal. Engineered features improved business interpretability more than raw predictive performance.

## 14. Deployment Artifact

- Final serving pipeline (`ColumnTransformer` + tuned `DecisionTreeClassifier`) persisted with `joblib` to `model/churn_pipeline.pkl`.
- This is the model loaded and served by [`app.py`](../app.py) via the `/predict` FastAPI endpoint (see [README.md](../README.md) for run/setup instructions).
