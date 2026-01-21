# 🎯 AI-Based Customer Churn Prediction for SaaS

A comprehensive machine learning solution for predicting and preventing customer churn in SaaS businesses, featuring advanced feature engineering, explainable AI, and an interactive dashboard for actionable insights.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Dashboard Features](#dashboard-features)
- [Business Impact](#business-impact)
- [Technical Details](#technical-details)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This project implements a production-ready churn prediction system that helps SaaS companies identify at-risk customers before they churn. The solution combines machine learning with explainable AI (SHAP) to not only predict churn but also explain *why* customers are likely to leave, enabling targeted retention strategies.

**Problem Statement**: Customer churn is one of the biggest challenges for SaaS businesses. This system helps companies:
- Identify high-risk customers early
- Understand the key drivers of churn
- Take proactive retention actions
- Optimize customer success strategies

## ✨ Key Features

### 1. **Advanced Feature Engineering**
- **Usage-based metrics**: Service adoption rate, total services used
- **Behavioral indicators**: Spending trends, price sensitivity
- **Risk flags**: High-risk contracts, risky payment methods
- **Customer lifecycle**: New customer flags, lifecycle stage segmentation
- **Value metrics**: Customer lifetime value (CLV), value per service

### 2. **High-Performance ML Models**
- **LightGBM Classifier**: Primary model with 74.3% AUC-ROC
- **Logistic Regression Baseline**: For comparison and interpretability
- **Class imbalance handling**: Built-in through scale_pos_weight
- **Time-aware validation**: Simulates real-world deployment scenarios

### 3. **Explainable AI (XAI)**
- **SHAP (SHapley Additive exPlanations)**: Global and local feature importance
- **Per-user explanations**: Top 3-5 risk factors for each customer
- **Business-friendly insights**: Translates technical findings into actionable recommendations

### 4. **Interactive Dashboard**
Built with Dash and Plotly, featuring:
- **A. Churn Risk Overview**: Distribution and tier analysis
- **B. Global Churn Drivers**: Top features driving churn across segments
- **C. User-Level Explanations**: Individual customer risk profiles
- **D. Business Insights**: Actionable recommendations and strategies

### 5. **Automated Reporting**
- Per-customer churn probability scores
- Risk tier classification (Low/Medium/High)
- Top 3 churn drivers for each customer
- Recommended retention actions
- Exportable CSV reports

## 📊 Dataset

**Source**: [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

**Description**: Contains 7,043 customer records with 21 features including:
- **Demographics**: Gender, SeniorCitizen, Partner, Dependents
- **Services**: PhoneService, MultipleLines, InternetService, OnlineSecurity, etc.
- **Account Info**: Contract type, PaymentMethod, PaperlessBilling
- **Financial**: MonthlyCharges, TotalCharges, tenure
- **Target**: Churn (Yes/No)

## 🚀 Installation

### Prerequisites
- Python 3.8+
- Google Colab (recommended) or local Jupyter environment
- Kaggle API credentials

### Setup Steps

1. **Clone or download the notebook**
```bash
git clone <your-repo-url>
cd saas-churn-prediction
```

2. **Install required packages**
```python
pip install pandas numpy scikit-learn xgboost lightgbm shap plotly dash==2.11.1 kaggle dash-bootstrap-components werkzeug==2.2.3
```

3. **Configure Kaggle API** (if running locally)
   - Download `kaggle.json` from your Kaggle account
   - Place it in `~/.kaggle/` directory
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

4. **Run in Google Colab** (recommended)
   - Upload the notebook to Colab
   - Follow the Kaggle authentication prompts
   - Run all cells sequentially

## 💻 Usage

### Running the Complete Pipeline

```python
# 1. Load and explore data
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 2. Engineer features
# (Automated in notebook cells 4-5)

# 3. Train model
lgbm = lgb.LGBMClassifier(...)
lgbm.fit(X_train, y_train)

# 4. Generate predictions
y_pred_proba = lgbm.predict_proba(X_test)[:, 1]

# 5. Launch dashboard
app.run_server(mode='inline', height=1200)

# 6. Export results
user_report.to_csv('churn_prediction_report.csv', index=False)
```

### Using the Dashboard

1. **Risk Overview**: View overall churn risk distribution
2. **Filter by Risk Tier**: Analyze High/Medium/Low risk segments
3. **Explore Drivers**: Identify top churn factors globally
4. **Customer Deep-Dive**: Select individual customers for detailed analysis
5. **Take Action**: Follow recommended retention strategies

## 📁 Project Structure

```
AI_based_Customer_churn_prediction(SAAS)_.ipynb
├── Section 1: Environment Setup
├── Section 2: Data Loading (Kaggle API)
├── Section 3: Data Understanding & EDA
├── Section 4: Feature Engineering
│   ├── Usage metrics
│   ├── Behavioral indicators
│   ├── Risk flags
│   └── Value metrics
├── Section 5: Model Training
│   ├── Baseline (Logistic Regression)
│   └── Advanced (LightGBM)
├── Section 6: Model Evaluation
│   ├── ROC Curve
│   └── Confusion Matrix
├── Section 7: Explainability (SHAP)
├── Section 8-9: Interactive Dashboard
└── Section 10: Report Generation
```

## 📈 Model Performance

### Metrics
- **AUC-ROC**: 0.743
- **Precision**: Optimized for business needs
- **Recall**: High detection of at-risk customers
- **F1-Score**: Balanced performance

### Key Insights
- **Top Churn Driver**: Contract type (Month-to-month = high risk)
- **Secondary Drivers**: ValuePerService, PriceSensitivity
- **Behavioral Signals**: ServiceAdoptionRate, SpendingTrend
- **Critical Period**: First 6 months (new customers)

## 🎛️ Dashboard Features

### A. Churn Risk Overview
- Histogram of churn probabilities
- Risk tier distribution (Low/Medium/High)
- Filterable by segment

### B. Global Churn Drivers
- Top 10 features by SHAP importance
- Segment-specific driver analysis
- Visual ranking of risk factors

### C. User-Level Explanations
- Individual customer profiles
- Top 5 risk factors per user
- Personalized retention recommendations
- Color-coded risk levels

### D. Business Insights
- High-level strategic recommendations
- Segment-specific action items
- ROI-focused interventions

## 💼 Business Impact

### Retention Strategy Framework

| Risk Tier | Action | Expected Impact |
|-----------|--------|-----------------|
| **High (>70%)** | URGENT: Retention call + contract upgrade offer | Save 30-40% of at-risk customers |
| **Medium (30-70%)** | Schedule check-in + product training | Reduce churn by 15-25% |
| **Low (<30%)** | Maintain engagement with value content | Prevent drift, increase LTV |

### ROI Calculation
- **Average Customer LTV**: $1,500
- **Churn Rate Reduction**: 20% (conservative)
- **Customers Saved per 1000**: 200
- **Annual Value**: $300,000

## 🔧 Technical Details

### Feature Engineering Highlights

```python
# Spending trajectory (churn signal if <1)
data['SpendingTrend'] = data['MonthlyCharges'] / (data['AvgMonthlySpend'] + 1)

# Service adoption density
data['ServiceAdoptionRate'] = data['TotalServicesUsed'] / (data['tenure'] + 1)

# Price sensitivity
data['PriceSensitivity'] = data['MonthlyCharges'] / (data['tenure'] + 1)
```

### Model Configuration

```python
lgbm = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    scale_pos_weight=2.77,  # Handles class imbalance
    subsample=0.8,
    colsample_bytree=0.8
)
```

### SHAP Integration

```python
explainer = shap.TreeExplainer(lgbm)
shap_values = explainer.shap_values(X_test)
# Per-user top drivers
top_drivers = np.argsort(np.abs(shap_values[user_idx]))[-5:]
```

## 🚀 Future Enhancements

1. **Model Improvements**
   - Ensemble methods (stacking)
   - Deep learning (LSTM for temporal patterns)
   - AutoML for hyperparameter optimization

2. **Feature Expansion**
   - Product usage logs (session frequency, feature adoption)
   - Customer support interactions
   - NPS scores and feedback sentiment

3. **Production Deployment**
   - Real-time prediction API
   - Automated alerting system
   - A/B testing framework for interventions

4. **Advanced Analytics**
   - Customer segmentation (RFM analysis)
   - Survival analysis (time-to-churn)
   - Causal inference (intervention impact)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: IBM Telco Customer Churn dataset via Kaggle
- **Libraries**: scikit-learn, LightGBM, SHAP, Plotly, Dash
- **Inspiration**: Real-world SaaS retention challenges

## 📧 Contact

For questions or collaboration:
- **LinkedIn**: [Abdullah Azhar ](https://www.linkedin.com/in/abdullah-azhar-009130308/)
- **GitHub**: [@AbddoesAI](https://github.com/AbddoesAI)

---

**⭐ If this project helps you, please star it on GitHub!**

*Built with ❤️ for the SaaS community*
