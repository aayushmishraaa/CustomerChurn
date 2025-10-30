
# Train and evaluate multiple ML models
print("="*80)
print("TRAINING MULTIPLE MACHINE LEARNING MODELS")
print("="*80)

# Initialize models with class weights to handle imbalanced data
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=42, max_depth=10),
    'Random Forest': RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, learning_rate=0.1),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Support Vector Machine': SVC(class_weight='balanced', probability=True, random_state=42, kernel='rbf')
}

# Store results
results = []
trained_models = {}

print("\nTraining models and evaluating performance...\n")

for name, model in models.items():
    print(f"Training {name}...")
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    if y_pred_proba is not None:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    else:
        roc_auc = np.nan
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    })
    
    # Save trained model
    trained_models[name] = model
    
    print(f"  ✓ Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    if not np.isnan(roc_auc):
        print(f"    ROC-AUC: {roc_auc:.4f}")
    print()

# Create results dataframe
results_df = pd.DataFrame(results)
results_df = results_df.round(4)
results_df = results_df.sort_values('F1-Score', ascending=False).reset_index(drop=True)

print("="*80)
print("MODEL PERFORMANCE COMPARISON")
print("="*80)
print(results_df.to_string(index=False))
print()

# Find best model
best_model_name = results_df.iloc[0]['Model']
best_model = trained_models[best_model_name]
print(f"\n🏆 Best Model: {best_model_name}")
print(f"   F1-Score: {results_df.iloc[0]['F1-Score']:.4f}")

# Save best model
with open('best_churn_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print(f"\n✓ Best model saved as 'best_churn_model.pkl'")

# Save all models
with open('all_models.pkl', 'wb') as f:
    pickle.dump(trained_models, f)

# Save results to CSV
results_df.to_csv('model_performance_results.csv', index=False)
print("✓ Performance results saved to 'model_performance_results.csv'")

print("\n" + "="*80)
