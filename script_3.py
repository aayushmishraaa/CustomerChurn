
# Build ML model without SMOTE - using class weights instead
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import pickle

print("="*80)
print("BUILDING MACHINE LEARNING MODELS FOR BANK CUSTOMER CHURN PREDICTION")
print("="*80)

# Data Preprocessing
print("\n1. DATA PREPROCESSING")
print("-" * 80)

# Create a copy of the dataframe
df_model = df.copy()

# Encode categorical variables
print("Encoding categorical variables...")
le_gender = LabelEncoder()
le_country = LabelEncoder()

df_model['gender_encoded'] = le_gender.fit_transform(df_model['gender'])
df_model['country_encoded'] = le_country.fit_transform(df_model['country'])

print(f"Gender mapping: {dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_)))}")
print(f"Country mapping: {dict(zip(le_country.classes_, le_country.transform(le_country.classes_)))}")

# Feature Engineering
print("\nCreating engineered features...")
# Age groups
df_model['age_group'] = pd.cut(df_model['age'], bins=[0, 30, 40, 50, 100], labels=[0, 1, 2, 3])
df_model['age_group'] = df_model['age_group'].astype(int)

# Balance categories
df_model['has_balance'] = (df_model['balance'] > 0).astype(int)
df_model['balance_category'] = pd.cut(df_model['balance'], bins=[-1, 0, 50000, 100000, 300000], labels=[0, 1, 2, 3])
df_model['balance_category'] = df_model['balance_category'].astype(int)

# Tenure categories
df_model['tenure_category'] = pd.cut(df_model['tenure'], bins=[-1, 2, 5, 7, 11], labels=[0, 1, 2, 3])
df_model['tenure_category'] = df_model['tenure_category'].astype(int)

# Credit score categories
df_model['credit_score_category'] = pd.cut(df_model['credit_score'], bins=[0, 400, 600, 700, 900], labels=[0, 1, 2, 3])
df_model['credit_score_category'] = df_model['credit_score_category'].astype(int)

# Interaction features
df_model['balance_to_salary_ratio'] = df_model['balance'] / (df_model['estimated_salary'] + 1)
df_model['products_per_tenure'] = df_model['products_number'] / (df_model['tenure'] + 1)
df_model['active_products_interaction'] = df_model['active_member'] * df_model['products_number']

print(f"Total engineered features created: {len(df_model.columns) - len(df.columns)}")

# Select features for modeling
feature_columns = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 
                  'credit_card', 'active_member', 'estimated_salary', 
                  'gender_encoded', 'country_encoded', 'age_group', 'has_balance',
                  'balance_category', 'tenure_category', 'credit_score_category',
                  'balance_to_salary_ratio', 'products_per_tenure', 'active_products_interaction']

X = df_model[feature_columns]
y = df_model['churn']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target variable shape: {y.shape}")
print(f"Class distribution: {dict(y.value_counts())}")

# Split the data
print("\n2. SPLITTING DATA (80% Train, 20% Test)")
print("-" * 80)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print(f"Training set churn rate: {y_train.mean()*100:.2f}%")
print(f"Test set churn rate: {y_test.mean()*100:.2f}%")

# Scale features
print("\n3. FEATURE SCALING")
print("-" * 80)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
    
# Save encoders
with open('encoders.pkl', 'wb') as f:
    pickle.dump({'gender': le_gender, 'country': le_country}, f)

print("Features scaled using StandardScaler and saved")

print("\n" + "="*80)
print("DATA PREPROCESSING COMPLETE")
print("="*80)
