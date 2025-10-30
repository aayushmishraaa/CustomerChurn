import plotly.graph_objects as go
import plotly.express as px

# Data from the provided JSON
features = ["Age", "Number of Products", "Active Products Interaction", "Active Member", "Balance", "Country", "Balance to Salary Ratio", "Age Group", "Credit Score", "Estimated Salary"]
importance = [37.21, 29.03, 6.99, 6.92, 4.65, 3.85, 2.75, 2.21, 1.87, 1.81]

# Create a gradient from dark green to light green
# Using a range of green colors from dark to light
green_colors = [
    '#1B4D3E',  # Dark green
    '#2E6B4F',
    '#4A8963',
    '#66A677',
    '#82C28B',
    '#9EDE9F',
    '#A8E6A8',
    '#B8EDB8',
    '#C8F4C8',
    '#D8FBD8'   # Light green
]

# Create horizontal bar chart
fig = go.Figure(data=[
    go.Bar(
        y=features,
        x=importance,
        orientation='h',
        marker=dict(color=green_colors),
        text=[f'{val}%' for val in importance],
        textposition='outside',
        textfont=dict(size=12)
    )
])

# Update layout
fig.update_layout(
    title="Top 10 Churn Features",
    xaxis_title="Importance %",
    yaxis_title="Features",
    yaxis=dict(categoryorder='array', categoryarray=features),
    showlegend=False
)

# Update traces for better appearance
fig.update_traces(cliponaxis=False)

# Update x-axis to show percentage format
fig.update_xaxes(range=[0, max(importance) * 1.1])

# Save as both PNG and SVG
fig.write_image("feature_importance_chart.png")
fig.write_image("feature_importance_chart.svg", format="svg")

fig.show()