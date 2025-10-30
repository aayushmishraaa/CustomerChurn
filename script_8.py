
# Create requirements.txt file
requirements_content = """pandas==1.5.3
numpy==1.24.3
scikit-learn==1.3.0
streamlit==1.28.0
matplotlib==3.7.2
seaborn==0.12.2
"""

with open('requirements.txt', 'w') as f:
    f.write(requirements_content)

print("✓ requirements.txt created successfully")

# Create a comprehensive project summary
print("\n" + "="*80)
print("PROJECT SUMMARY: BANK CUSTOMER CHURN PREDICTION")
print("="*80)

print("\n📦 DELIVERABLES:")
print("-" * 80)
deliverables = [
    "1. Trained Machine Learning Models (6 algorithms)",
    "2. Best Model: Gradient Boosting (86.95% accuracy, 60.39% F1-score)",
    "3. Streamlit Web Application (streamlit_app.py)",
    "4. Model Files: best_churn_model.pkl, all_models.pkl",
    "5. Preprocessing Files: scaler.pkl, encoders.pkl",
    "6. Performance Analysis: model_performance_results.csv",
    "7. Feature Importance: feature_importance.csv",
    "8. EDA Visualizations: churn_eda_analysis.png",
    "9. Complete Documentation: README.md",
    "10. Dependencies: requirements.txt"
]

for item in deliverables:
    print(f"  {item}")

print("\n📊 KEY STATISTICS:")
print("-" * 80)
print(f"  • Total Customers: 10,000")
print(f"  • Churned: 2,037 (20.37%)")
print(f"  • Retained: 7,963 (79.63%)")
print(f"  • Features Used: 18 (including engineered features)")
print(f"  • Training Set: 8,000 samples (80%)")
print(f"  • Test Set: 2,000 samples (20%)")

print("\n🎯 MODEL PERFORMANCE:")
print("-" * 80)
print("  Best Model: Gradient Boosting Classifier")
print(f"  • Accuracy: 86.95%")
print(f"  • Precision: 78.97%")
print(f"  • Recall: 48.89%")
print(f"  • F1-Score: 60.39%")
print(f"  • ROC-AUC: 86.87%")

print("\n🔑 TOP 5 IMPORTANT FEATURES:")
print("-" * 80)
print("  1. Age (37.21%)")
print("  2. Number of Products (29.03%)")
print("  3. Active Products Interaction (6.99%)")
print("  4. Active Member Status (6.92%)")
print("  5. Balance (4.65%)")

print("\n🌐 WEB APPLICATION:")
print("-" * 80)
print("  • Framework: Streamlit")
print("  • Features: Interactive prediction interface")
print("  • Real-time predictions with probability scores")
print("  • Custom CSS styling")
print("  • User-friendly design")

print("\n🚀 HOW TO RUN:")
print("-" * 80)
print("  1. Install dependencies:")
print("     pip install -r requirements.txt")
print()
print("  2. Run the Streamlit app:")
print("     streamlit run streamlit_app.py")
print()
print("  3. Open browser at: http://localhost:8501")

print("\n💡 KEY INSIGHTS:")
print("-" * 80)
print("  • Female customers churn 25% vs 16.45% for males")
print("  • German customers have highest churn rate (32.44%)")
print("  • Customers with 3-4 products more likely to churn")
print("  • Inactive members significantly more likely to leave")
print("  • Age is the strongest predictor of churn")

print("\n" + "="*80)
print("PROJECT COMPLETED SUCCESSFULLY! 🎉")
print("="*80)
