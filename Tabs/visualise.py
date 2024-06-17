import warnings 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn import tree
import streamlit as st

from web_function import load_data, train_model, predict

def app(df, x, y):

    warnings.filterwarnings("ignore")
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.title("Halaman Visualisasi")

    if st.checkbox("Tampilkan Data"):
        model, score = train_model(x, y)
        y_pred = model.predict(x)
        cm = confusion_matrix(y, y_pred)
        
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["ckd", "notckd"], yticklabels=["ckd", "notckd"])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        st.pyplot()
    
    if st.checkbox("Tampilkan Decision Tree"):
        model, score = train_model(x, y)
        dot_data = tree.export_graphviz(model, out_file=None, 
                                        filled=True, rounded=True, 
                                        feature_names=x.columns, 
                                        class_names=["ckd", "notckd"])
        st.graphviz_chart(dot_data)

# You can now call this function with your data
# df = load_data() # This should be your data loading function
# x, y = df.drop(columns=['target']), df['target'] # Modify this based on your actual data structure
# app(df, x, y)
