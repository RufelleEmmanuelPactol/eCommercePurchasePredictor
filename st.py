import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings

# Set Streamlit theme to vibrant and neon
st.set_page_config(layout="wide")

# Set a custom color palette
custom_palette = sns.color_palette(["#FF5733", "#FFC300", "#DAF7A6", "#C70039", "#900C3F"])

# Load and preprocess the data
@st.cache
def load_data():
    df = pd.read_csv('archive/online_shoppers_intention.csv')
    X = df[['Administrative', 'Administrative_Duration', 'ProductRelated', 'SpecialDay', 'ProductRelated_Duration']]
    y = df['Revenue']
    return df, X, y

df, X, y = load_data()

# Sidebar for user input
st.sidebar.header("Model Parameters")
test_size = st.sidebar.slider("Test Size", 0.1, 0.3, 0.15, key='test_size_slider')

# Display the dataframe
st.subheader("ECommerce Prediction With Logistic Regression")

st.write("Please be patient as the data is computationally intensive. Graphs may take a while to load longer than usual.")
# EDA
st.subheader("Exploratory Data Analysis")
independents = ['Administrative', 'Administrative_Duration', 'ProductRelated', 'SpecialDay']
eda = df[independents + ['Revenue']]
sns.set(style="whitegrid")  # Set Seaborn style
fig_eda = sns.pairplot(eda, hue='Revenue', palette=custom_palette).fig
st.pyplot(fig_eda)

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=55)
model = LogisticRegression()
model.fit(X_train, y_train)
y_hat = model.predict(X_test)
score = accuracy_score(y_test, y_hat)
y_probs = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Accuracy
st.metric("Accuracy Score", score)

# User input for prediction
st.subheader("Make a Prediction")
admin = st.number_input("Administrative (number of pages visited)", min_value=0)
admin_duration = st.number_input("Administrative Duration (time in seconds)", min_value=0.0, format="%.2f")
product_related = st.number_input("ProductRelated (number of viewed pages with related products)", min_value=0)
special_day = st.number_input("SpecialDay (number of days prior to a special day)", min_value=0.0, format="%.2f")
product_related_duration = st.number_input("ProductRelated Duration (time in seconds)", min_value=0.0, format="%.2f")

if st.button('Predict'):
    prediction = model.predict([[admin, admin_duration, product_related, special_day, product_related_duration]])
    prediction_proba = model.predict_proba([[admin, admin_duration, product_related, special_day, product_related_duration]])
    st.write(f"Prediction: {'Revenue Generation Likely' if prediction[0] else 'Revenue Generation Unlikely'}")
    st.write(f"Prediction Probability: {prediction_proba[0][1]:.2f}")

# ROC Curve
st.subheader("ROC Curve")
fig, ax = plt.subplots()
ax.plot(fpr, tpr, color='#FF5733', lw=3, label='ROC curve (area = %0.2f)' % roc_auc)  # Neon orange
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic')
ax.legend(loc="lower right")
st.pyplot(fig)

st.subheader("S-Curve Visualization")

# Define percentiles for outlier removal
lower_percentile = 5
upper_percentile = 95

# Calculate percentile values
lower_bound = np.percentile(y_probs, lower_percentile)
upper_bound = np.percentile(y_probs, upper_percentile)

# Create a mask to filter out outliers
mask = (y_probs >= lower_bound) & (y_probs <= upper_bound)

# Apply mask to probabilities and actual outcomes
filtered_probs = y_probs[mask]
filtered_actuals = y_test.to_numpy()[mask]

# Plotting
fig_s_curve, ax_s_curve = plt.subplots(figsize=(8, 6))
ax_s_curve.scatter(filtered_probs, filtered_actuals, alpha=0.2)
ax_s_curve.axhline(y=0.15, color='red', linestyle='--')
ax_s_curve.set_xlabel('Predicted Probability')
ax_s_curve.set_ylabel('Actual Outcome')
ax_s_curve.set_title('Logistic Regression S-Curve Visualization (Outliers Removed)')
st.pyplot(fig_s_curve)

# Find the optimal threshold (closest to top-left corner of ROC curve)
optimal_idx = np.argmax(true_positive - false_positive)
optimal_threshold = thresholds[optimal_idx]

# Apply the threshold to determine the state
states = np.where(y_probs > optimal_threshold, 'True', 'False')

# Visualize the result
fig_threshold_visualization, ax_threshold_visualization = plt.subplots(figsize=(10, 6))
for state in np.unique(states):
    mask = states == state
    ax_threshold_visualization.scatter(np.arange(len(y_probs))[mask], y_probs[mask],
                                       color=colors[state], label=f'{state} (Threshold = {optimal_threshold:.2f})', alpha=0.3)

ax_threshold_visualization.set_xlabel('Sorted Index')
ax_threshold_visualization.set_ylabel('Predicted Probability')
ax_threshold_visualization.set_title('Logistic Regression Probability Threshold Visualization')
ax_threshold_visualization.legend()
st.pyplot(fig_threshold_visualization)

