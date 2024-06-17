import streamlit as st

from web_function import load_data, train_model, predict

def app(df,x,y):

    st.title("Halaman Prediksi")

    col1,col2,col3 = st.columns(3)

    with col1:
        bp = st.text_input("Input nilai bp")
    with col1:
        sg = st.text_input("Input nilai sg")
    with col1:
        al = st.text_input("Input nilai al")
    with col1:
        su = st.text_input("Input nilai su")
    with col1:
        rbc = st.text_input("Input nilai rbc")
    with col1:
        pc = st.text_input("Input nilai pc")
    with col1:
        pcc = st.text_input("Input nilai pcc")
    with col1:
        ba = st.text_input("Input nilai ba")
    with col2:
        bgr = st.text_input("Input nilai bgr")
    with col2:
        bu = st.text_input("Input nilai bu")
    with col2:
        sc = st.text_input("Input nilai sc")
    with col2:
        sod = st.text_input("Input nilai sod")
    with col2:
        pot = st.text_input("Input nilai pot")
    with col2:
        hemo = st.text_input("Input nilai hemo")
    with col2:
        pcv = st.text_input("Input nilai pcv")
    with col2:
        wc = st.text_input("Input nilai wc")
    with col3:
        rc = st.text_input("Input nilai rc")
    with col3:
        htn = st.text_input("Input nilai htn")
    with col3:
        dm = st.text_input("Input nilai dm")
    with col3:
        cad = st.text_input("Input nilai cad")
    with col3:
        appet = st.text_input("Input nilai appet")
    with col3:
        pe = st.text_input("Input nilai pe")
    with col3:
        ane = st.text_input("Input nilai ane")


    features = [bp,sg,al,su,rbc,pc,pcc,ba,bgr,bu,sc,sod,pot,hemo,pcv,wc,rc,htn,dm,cad,appet,pe,ane]

    if st.button("Predict"):
        prediction, score = predict(x,y,features)
        score = score
        st.info("Prediksi Sukses")

        if(prediction == 1):
            st.warning("Pasien terkena penyakit ginjal")
        else:
            st.success("Pasien tidak terkena penyakit ginjal")

        st.write("Score model : ",(score*100),"%")
    
