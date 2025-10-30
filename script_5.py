
# Detailed evaluation of the best model
print("="*80)
print(f"DETAILED EVALUATION: {best_model_name}")
print("="*80)

# Predictions
y_pred = best_model.predict(X_test_scaled)
y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print("-" * 40)
print(f"                Predicted")
print(f"              No Churn  Churn")
print(f"Actual No    {cm[0][0]:8d}  {cm[0][1]:5d}")
print(f"       Yes   {cm[1][0]:8d}  {cm[1][1]:5d}")

# Classification Report
print("\n" + "-"*80)
print("Classification Report:")
print("-" * 80)
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'], digits=4))

# Calculate additional metrics
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
npv = tn / (tn + fn)

print("-" * 80)
print("Additional Metrics:")
print("-" * 80)
print(f"True Negatives:  {tn:4d}  |  True Positives:   {tp:4d}")
print(f"False Negatives: {fn:4d}  |  False Positives:  {fp:4d}")
print(f"\nSpecificity (True Negative Rate): {specificity:.4f}")
print(f"Negative Predictive Value:        {npv:.4f}")

# Feature Importance (for Gradient Boosting)
if hasattr(best_model, 'feature_importances_'):
    print("\n" + "="*80)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("="*80)
    
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(feature_importance.head(10).to_string(index=False))
    
    # Save feature importance
    feature_importance.to_csv('feature_importance.csv', index=False)
    print("\n✓ Feature importance saved to 'feature_importance.csv'")

print("\n" + "="*80)
print("MODEL EVALUATION COMPLETE")
print("="*80)
