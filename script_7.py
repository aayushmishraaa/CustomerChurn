
# Create README documentation for the project
readme_content = """# Bank Customer Churn Prediction - Machine Learning Project

## 📋 Project Overview

This is a comprehensive end-to-end machine learning project that predicts whether a bank customer will churn (leave the bank) based on various customer attributes. The project includes data exploration, feature engineering, multiple model training, evaluation, and a deployed web application using Streamlit.

## 🎯 Business Problem

Customer churn is a critical challenge in the banking sector, where retaining existing customers is significantly more cost-effective than acquiring new ones. This project aims to:
- Predict customers at risk of churning
- Enable proactive retention strategies
- Reduce customer attrition rates
- Improve customer satisfaction and loyalty

## 📊 Dataset Information

The dataset contains 10,000 bank customer records with the following features:

### Features:
- **customer_id**: Unique identifier for each customer
- **credit_score**: Customer's credit score (350-850)
- **country**: Customer's country (France, Germany, Spain)
- **gender**: Customer's gender (Male, Female)
- **age**: Customer's age (18-92)
- **tenure**: Years as a bank customer (0-10)
- **balance**: Account balance
- **products_number**: Number of bank products used (1-4)
- **credit_card**: Whether customer has a credit card (0/1)
- **active_member**: Whether customer is an active member (0/1)
- **estimated_salary**: Customer's estimated salary
- **churn**: Target variable - whether customer churned (0/1)

### Dataset Statistics:
- Total Customers: 10,000
- Churned Customers: 2,037 (20.37%)
- Retained Customers: 7,963 (79.63%)
- Class Imbalance: Yes (handled using class weights)

## 🔧 Technical Stack

### Programming Language:
- Python 3.8+

### Libraries Used:
- **Data Analysis**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn
- **Model Serialization**: pickle
- **Web Application**: Streamlit

## 🏗️ Project Structure

```
bank-churn-prediction/
│
├── Bank-Customer-Churn-Prediction.csv    # Dataset
├── streamlit_app.py                      # Streamlit web application
├── best_churn_model.pkl                  # Trained Gradient Boosting model
├── all_models.pkl                        # All trained models
├── scaler.pkl                            # Feature scaler
├── encoders.pkl                          # Label encoders for categorical variables
├── model_performance_results.csv         # Model comparison results
├── feature_importance.csv                # Feature importance scores
├── churn_eda_analysis.png               # EDA visualizations
├── requirements.txt                      # Python dependencies
└── README.md                            # Project documentation
```

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8 or higher installed on your system.

### Installation

1. Clone the repository or download the project files
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

### Requirements.txt Contents:

```
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.3.0
streamlit==1.28.0
matplotlib==3.7.2
seaborn==0.12.2
```

## 📈 Machine Learning Pipeline

### 1. Data Preprocessing
- Handled categorical variables using Label Encoding
- Created engineered features (age groups, balance categories, interaction features)
- Scaled features using StandardScaler
- Split data: 80% training, 20% testing
- Maintained stratification to preserve class distribution

### 2. Feature Engineering
Created multiple engineered features:
- **age_group**: Categorized age into 4 groups
- **has_balance**: Binary indicator for positive balance
- **balance_category**: Categorized balance into 4 ranges
- **tenure_category**: Categorized tenure into 4 groups
- **credit_score_category**: Categorized credit scores
- **balance_to_salary_ratio**: Financial health indicator
- **products_per_tenure**: Product adoption rate
- **active_products_interaction**: Engagement metric

### 3. Models Trained

Six different machine learning models were trained and evaluated:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Gradient Boosting | 86.95% | 78.97% | 48.89% | 60.39% | 86.87% |
| Support Vector Machine | 77.80% | 47.14% | 74.94% | 57.87% | 84.52% |
| Random Forest | 85.90% | 78.54% | 42.26% | 54.95% | 85.25% |
| Decision Tree | 74.70% | 42.11% | 64.86% | 51.06% | 74.17% |
| Logistic Regression | 69.75% | 37.04% | 69.53% | 48.33% | 78.12% |
| K-Nearest Neighbors | 83.30% | 65.94% | 37.10% | 47.48% | 76.63% |

**Best Model**: Gradient Boosting Classifier
- Highest F1-Score: 60.39%
- Best balance between precision and recall
- ROC-AUC Score: 86.87%

### 4. Model Evaluation

**Confusion Matrix (Gradient Boosting)**:
```
                Predicted
              No Churn  Churn
Actual No        1540     53
       Yes        208    199
```

**Performance Metrics**:
- True Negatives: 1540
- True Positives: 199
- False Negatives: 208
- False Positives: 53
- Specificity: 96.67%
- Negative Predictive Value: 88.10%

### 5. Feature Importance

Top 10 Most Important Features (Gradient Boosting):
1. Age (37.21%)
2. Number of Products (29.03%)
3. Active Products Interaction (6.99%)
4. Active Member Status (6.92%)
5. Balance (4.65%)
6. Country (3.85%)
7. Balance to Salary Ratio (2.75%)
8. Age Group (2.21%)
9. Credit Score (1.87%)
10. Estimated Salary (1.81%)

## 🌐 Web Application

### Features:
- Interactive user interface for churn prediction
- Real-time predictions based on customer inputs
- Probability scores for churn likelihood
- Customer profile summary
- Responsive design with custom styling

### Running the Application:

```bash
streamlit run streamlit_app.py
```

The app will open in your default web browser at `http://localhost:8501`

### Application Workflow:
1. User enters customer information (demographics, account details)
2. App performs feature engineering automatically
3. Model predicts churn probability
4. Results displayed with risk level and recommendations

## 📊 Key Insights from EDA

1. **Gender**: Female customers have higher churn rate (25%) compared to males (16.45%)
2. **Geography**: German customers show highest churn rate (32.44%)
3. **Age**: Older customers are more likely to churn
4. **Products**: Customers with 3-4 products show higher churn rates
5. **Active Membership**: Inactive members are significantly more likely to churn
6. **Balance**: Customers with very high or zero balance show higher churn

## 🎯 Business Recommendations

Based on model insights:
1. **Focus on product optimization**: Avoid forcing multiple products
2. **Enhance engagement**: Target inactive members with personalized campaigns
3. **Age-specific strategies**: Special retention programs for older customers
4. **Geographic customization**: Tailor services for German market
5. **Female customer retention**: Develop targeted programs for female customers

## 🔮 Future Improvements

1. **Deep Learning**: Implement neural networks for potentially better performance
2. **Real-time predictions**: Connect to live customer database
3. **A/B Testing**: Test retention strategies on predicted churners
4. **Feature expansion**: Include transaction history and customer service interactions
5. **Automated retraining**: Set up pipeline for regular model updates
6. **Explainability**: Implement SHAP values for better interpretability
7. **Cost-sensitive learning**: Optimize for business value, not just accuracy

## 📝 How to Use This Project

### For Training:
```python
# Load and preprocess data
# Train models
# Evaluate performance
# Save best model
```

### For Prediction:
```python
import pickle
import numpy as np

# Load model
with open('best_churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make prediction
prediction = model.predict(features)
```

### For Deployment:
```bash
# Run Streamlit app
streamlit run streamlit_app.py
```

## 📚 References

1. Research papers on customer churn prediction in banking
2. Scikit-learn documentation
3. Streamlit documentation
4. Machine learning best practices

## 👨‍💻 Author

Machine Learning Project - Bank Customer Churn Prediction

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- Dataset sourced from banking industry
- Inspired by various churn prediction research papers
- Built with open-source libraries and frameworks

---

**Note**: This project is for educational and demonstration purposes. For production use, additional validation, testing, and regulatory compliance checks would be required.
"""

# Save README
with open('README.md', 'w') as f:
    f.write(readme_content)

print("✓ README.md created successfully")
