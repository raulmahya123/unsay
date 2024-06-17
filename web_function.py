import pandas as pd
from sklearn.preprocessing import LabelEncoder
import streamlit as st

@st.cache_data
def load_data():
    df = pd.read_csv('kidney_disease.csv')

    # Define the categorical columns
    categorical_cols = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]

    # Initialize label encoders for each categorical column
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))  # Ensure all data is treated as string
        label_encoders[col] = le

    x = df[["bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc", "htn", "dm", "cad", "appet", "pe", "ane"]]
    y = df["classification"].apply(lambda x: 1 if x == 'ckd' else 0)

    return df, x, y, label_encoders
