import pandas as pd
import streamlit as st

st.set_page_config(page_title="MovieIQ", layout="wide")
st.title("🎬 MovieIQ — Predicting Movie Success")

df = pd.read_csv('movies.csv')
st.write("Data loaded:", df.shape)
import ast

df['success'] = (df['revenue'] > df['budget']).astype(int)

def extract_genres(genre_str):
    try:
        genre_list = ast.literal_eval(genre_str)
        return [g['name'] for g in genre_list]
    except (ValueError, SyntaxError):
        return []

df['genre_list'] = df['genres'].apply(extract_genres)
genre_df = df.explode('genre_list')

st.sidebar.header("Filter Movies")
all_genres = sorted(genre_df['genre_list'].dropna().unique())
selected_genre = st.sidebar.selectbox("Select Genre", ["All"] + all_genres)
min_vote = st.sidebar.slider("Minimum Vote Average", 0.0, 10.0, 0.0, 0.1)

filtered_df = df.copy()
if selected_genre != "All":
    filtered_df = filtered_df[filtered_df['genre_list'].apply(lambda genres: selected_genre in genres)]
filtered_df = filtered_df[filtered_df['vote_average'] >= min_vote]

st.write(f"Showing {filtered_df.shape[0]} movies matching filters")
import ast

df['success'] = (df['revenue'] > df['budget']).astype(int)

def extract_genres(genre_str):
    try:
        genre_list = ast.literal_eval(genre_str)
        return [g['name'] for g in genre_list]
    except (ValueError, SyntaxError):
        return []

df['genre_list'] = df['genres'].apply(extract_genres)
genre_df = df.explode('genre_list')
import matplotlib.pyplot as plt
import seaborn as sns

st.subheader("Budget vs Revenue")
plt.figure(figsize=(8,6))
plt.scatter(df['budget'], df['revenue'], alpha=0.4)
plt.xlabel('Budget')
plt.ylabel('Revenue')
st.pyplot(plt)

st.subheader("Most Common Genres")
st.bar_chart(genre_df['genre_list'].value_counts())

st.subheader("Success Rate by Genre (%)")
genre_success = genre_df.groupby('genre_list')['success'].mean().sort_values(ascending=False) * 100
st.bar_chart(genre_success)

st.subheader("Correlation Heatmap")
numeric_cols = ['budget', 'revenue', 'popularity', 'runtime', 'vote_average']
plt.figure(figsize=(8,6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
st.pyplot(plt)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

features = ['budget', 'popularity', 'runtime', 'vote_average']
X = df[features]
y = df['success']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

st.subheader("Model Performance")
st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.write("Precision:", precision_score(y_test, y_pred))
st.write("Recall:", recall_score(y_test, y_pred))

st.subheader("Predict a Movie's Success")
input_budget = st.number_input("Budget", min_value=0, value=50000000)
input_popularity = st.number_input("Popularity", min_value=0.0, value=50.0)
input_runtime = st.number_input("Runtime (minutes)", min_value=0, value=120)
input_vote = st.number_input("Vote Average", min_value=0.0, max_value=10.0, value=6.5)

if st.button("Predict"):
    input_data = pd.DataFrame([[input_budget, input_popularity, input_runtime, input_vote]], columns=features)
    prediction = rf_model.predict(input_data)[0]
    if prediction == 1:
        st.success("Predicted: SUCCESS 🎉")
    else:
        st.error("Predicted: NOT SUCCESSFUL")