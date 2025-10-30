import plotly.graph_objects as go
import numpy as np

# Data from the provided JSON
models = ["Gradient Boosting", "Support Vector Machine", "Random Forest", "Decision Tree", "Logistic Regression", "K-Nearest Neighbors"]
f1_scores = [0.6039, 0.5787, 0.5495, 0.5106, 0.4833, 0.4748]

# Abbreviate model names to fit 15 character limit for labels
model_labels = ["Gradient Boost", "SVM", "Random Forest", "Decision Tree", "Logistic Reg", "KNN"]

# Create color gradient from light to dark blue based on F1-score values
# Normalize scores to create gradient
min_score = min(f1_scores)
max_score = max(f1_scores)
normalized_scores = [(score - min_score) / (max_score - min_score) for score in f1_scores]

# Create blue gradient colors (light to dark blue)
colors = [f'rgba({int(173 - 100*norm)}, {int(216 - 80*norm)}, {int(230 - 50*norm)}, 1)' for norm in normalized_scores]

# Create horizontal bar chart
fig = go.Figure(go.Bar(
    x=f1_scores,
    y=model_labels,
    orientation='h',
    marker=dict(color=colors),
    text=[f'{score:.4f}' for score in f1_scores],
    textposition='inside',
    textfont=dict(color='white', size=12)
))

# Update layout
fig.update_layout(
    title="ML Model Performance (F1-Score)",
    xaxis_title="F1-Score",
    yaxis_title="Model",
    yaxis=dict(categoryorder='total ascending')
)

# Update traces for better appearance
fig.update_traces(cliponaxis=False)

# Save as both PNG and SVG
fig.write_image("chart.png")
fig.write_image("chart.svg", format="svg")

print("Chart saved successfully as chart.png and chart.svg")