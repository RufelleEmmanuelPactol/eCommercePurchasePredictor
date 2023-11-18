import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings

sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'font.family': 'Arial'})


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
df = pd.read_csv('archive/online_shoppers_intention.csv')
df.head()
# Let's identify the important features.
# Administrative - number of pages visited
# Administrative_Duration - time units (s) the user has been in the page
# ProductRelated - number of viewed pages with related products
# SpecialDay - number of days prior to a rather special day (Mother's Day, etc...)
X = df[['Administrative', 'Administrative_Duration', 'ProductRelated', 'SpecialDay'
    , 'ProductRelated_Duration']]
# Revenue - Output
y = df['Revenue']
# Updated cache decorator
@st.cache_data
def load_data():
    df = pd.read_csv('archive/online_shoppers_intention.csv')
    X = df[['Administrative', 'Administrative_Duration', 'ProductRelated', 'SpecialDay', 'ProductRelated_Duration']]
    y = df['Revenue']
    return df, X, y

# Let's split this data! :) Let's goo!!!
# Initializing the model!!!! 0v0
# Let's split this data! :) Let's goo!!!

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=55)
model = LogisticRegression()
model.fit(X_train, y_train)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=55)

warnings.filterwarnings("ignore")

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
test_size = st.sidebar.slider("Test Size", 0.1, 0.3, 0.15)

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=55)
model = LogisticRegression()
model.fit(X_train, y_train)
y_hat = model.predict(X_test)
score = accuracy_score(y_test, y_hat)
y_probs = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")

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
st.write("Data Preview:", df.head())

# EDA
st.subheader("Exploratory Data Analysis")
independents = ['Administrative', 'Administrative_Duration', 'ProductRelated', 'SpecialDay']
eda = df[independents + ['Revenue']]
fig_eda = sns.pairplot(eda, hue='Revenue').fig
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
ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic')
ax.legend(loc="lower right")
st.pyplot(fig)

# Accuracy
st.metric("Accuracy Score", score)

# S-Curve Visualization
st.subheader("S-Curve Visualization")
plt.figure(figsize=(8, 6))
plt.scatter(y_probs, y_test, alpha=0.2)
plt.axhline(y=0.5, color='red', linestyle='--')
plt.xlabel('Predicted Probability')
plt.ylabel('Actual Outcome')
plt.title('Logistic Regression S-Curve Visualization')
st.pyplot(plt)

# Probability Threshold Visualization
st.subheader("Probability Threshold Visualization")
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
states = np.where(y_probs > optimal_threshold, 'True', 'False')
fig, ax = plt.subplots(figsize=(10, 6))
colors = {'True': 'blue', 'False': 'red'}
for state in np.unique(states):
    mask = states == state
    ax.scatter(np.arange(len(y_probs))[mask], y_probs[mask], color=colors[state], label=f'{state} (Threshold = {optimal_threshold:.2f})', alpha=0.3)
ax.set_xlabel('Sorted Index')
ax.set_ylabel('Predicted Probability')
ax.set_title('Logistic Regression Probability Threshold Visualization')
ax.legend()
st.pyplot(fig)
