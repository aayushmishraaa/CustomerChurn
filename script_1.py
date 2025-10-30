
# Perform comprehensive EDA and create visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 1. Create churn distribution analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Churn distribution
churn_counts = df['churn'].value_counts()
axes[0, 0].bar(['Retained (0)', 'Churned (1)'], churn_counts.values, color=['#2ecc71', '#e74c3c'])
axes[0, 0].set_title('Customer Churn Distribution', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Number of Customers')
for i, v in enumerate(churn_counts.values):
    axes[0, 0].text(i, v + 100, str(v), ha='center', va='bottom', fontweight='bold')

# Age distribution by churn
df[df['churn']==0]['age'].hist(bins=30, alpha=0.6, label='Retained', ax=axes[0, 1], color='#2ecc71')
df[df['churn']==1]['age'].hist(bins=30, alpha=0.6, label='Churned', ax=axes[0, 1], color='#e74c3c')
axes[0, 1].set_title('Age Distribution by Churn Status', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Age')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()

# Gender vs Churn
gender_churn = pd.crosstab(df['gender'], df['churn'], normalize='index') * 100
gender_churn.plot(kind='bar', ax=axes[1, 0], color=['#2ecc71', '#e74c3c'])
axes[1, 0].set_title('Churn Rate by Gender', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Gender')
axes[1, 0].set_ylabel('Percentage (%)')
axes[1, 0].legend(['Retained', 'Churned'])
axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=0)

# Country vs Churn
country_churn = pd.crosstab(df['country'], df['churn'], normalize='index') * 100
country_churn.plot(kind='bar', ax=axes[1, 1], color=['#2ecc71', '#e74c3c'])
axes[1, 1].set_title('Churn Rate by Country', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Country')
axes[1, 1].set_ylabel('Percentage (%)')
axes[1, 1].legend(['Retained', 'Churned'])
axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=0)

plt.tight_layout()
plt.savefig('churn_eda_analysis.png', dpi=300, bbox_inches='tight')
print("EDA visualization saved!")

# Calculate churn statistics
print("\n" + "="*80)
print("CHURN STATISTICS BY CATEGORIES")
print("="*80)

print("\n1. Churn Rate by Gender:")
gender_stats = df.groupby('gender')['churn'].agg(['sum', 'count', 'mean'])
gender_stats.columns = ['Churned', 'Total', 'Churn_Rate']
gender_stats['Churn_Rate'] = (gender_stats['Churn_Rate'] * 100).round(2)
print(gender_stats)

print("\n2. Churn Rate by Country:")
country_stats = df.groupby('country')['churn'].agg(['sum', 'count', 'mean'])
country_stats.columns = ['Churned', 'Total', 'Churn_Rate']
country_stats['Churn_Rate'] = (country_stats['Churn_Rate'] * 100).round(2)
print(country_stats)

print("\n3. Churn Rate by Number of Products:")
products_stats = df.groupby('products_number')['churn'].agg(['sum', 'count', 'mean'])
products_stats.columns = ['Churned', 'Total', 'Churn_Rate']
products_stats['Churn_Rate'] = (products_stats['Churn_Rate'] * 100).round(2)
print(products_stats)

print("\n4. Churn Rate by Active Membership:")
active_stats = df.groupby('active_member')['churn'].agg(['sum', 'count', 'mean'])
active_stats.columns = ['Churned', 'Total', 'Churn_Rate']
active_stats['Churn_Rate'] = (active_stats['Churn_Rate'] * 100).round(2)
active_stats.index = ['Inactive', 'Active']
print(active_stats)

print("\n5. Average Values by Churn Status:")
numerical_cols = ['credit_score', 'age', 'tenure', 'balance', 'estimated_salary']
churn_comparison = df.groupby('churn')[numerical_cols].mean().round(2)
churn_comparison.index = ['Retained', 'Churned']
print(churn_comparison)
