import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

@st.cache_data
def load_data():
    df = pd.read_excel("Cleaned_DS_DATASET.xlsx")
    return df

df = load_data()

st.title("Nykaa Product Recommendation Dashboard")

st.header("Product Recommendation System")

@st.cache_resource
def create_tfidf_matrix(data):
    rec_vec = TfidfVectorizer(stop_words="english")
    tfidf_matrix = rec_vec.fit_transform(
        data['product description'].fillna('') + ' ' + data['product tags'].fillna('')
    )
    return rec_vec, tfidf_matrix

rec_vectorizer, tfidf_matrix = create_tfidf_matrix(df)

product_name = st.text_input("Enter a product name:", "Lakme Lipstick")

if st.button("Find Similar Products"):
    if product_name:
        try:
            idx = df[df['product name'].str.lower() == product_name.lower()].index[0]
            cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
            similar_indices = cosine_sim.argsort()[-6:][::-1][1:]
            similar_products = df.iloc[similar_indices][['product name', 'product brand', 'product rating']]
            st.write("Top 5 Similar Products:")
            st.dataframe(similar_products)
        except:
            st.error("Product not found. Please try another name.")
