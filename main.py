import streamlit as st
from web_function import load_data, train_model, predict

from Tabs import home, predict, visualise

Tabs = {
    "Home": home,
    "Predict": predict,
    "Visualise": visualise
}

st.sidebar.title("Navigasi")

page = st.sidebar.radio("pages", tuple(Tabs.keys()))

# Ensure load_data is called correctly
df, x, y = load_data()

if page in ["Predict", "Visualise"]:
    Tabs[page].app(df, x, y)
else:
    Tabs[page].app()
