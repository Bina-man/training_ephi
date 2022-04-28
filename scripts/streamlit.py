import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def Introduction():
    st.subheader("Diabetes Data")
    df = pd.read_csv("../data/diabetes.csv")
    df_copy = df.copy()
    df_copy.columns = [c.replace(" ", "_") for c in df_copy.columns]
    df_copy.rename(columns={"Outcome": "Has Diabetese"}, inplace=True)    
    df_copy['Has Diabetese'] = df_copy['Has Diabetese'].map({0: 'No', 1: 'Yes'})
    st.dataframe(df_copy.head())
    st.subheader("Data Visualization")
    # plot histogram
    st.subheader("Histogram")
    # plot histogram for all the columns 
    st.set_option('deprecation.showPyplotGlobalUse', False)
    choice = st.selectbox("Select Column", df_copy.columns)
    if choice == "Pregnancies":
        st.subheader("Histogram")
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(111)
        ax.hist(df_copy['Pregnancies'], bins=20)
        ax.set_title("Pregnancies")
        ax.set_xlabel("Pregnancies")
        ax.set_ylabel("Frequency")
        st.pyplot()
    elif choice == "Glucose":
        st.subheader("Histogram")
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(111)
        ax.hist(df_copy['Glucose'], bins=20)
        ax.set_title("Glucose")
        ax.set_xlabel("Glucose")
        ax.set_ylabel("Frequency")
        st.pyplot()
    elif choice == "BloodPressure":
        st.subheader("Histogram")
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(111)
        ax.hist(df_copy['BloodPressure'], bins=20)
        ax.set_title("Blood_Pressure")
        ax.set_xlabel("Blood_Pressure")
        ax.set_ylabel("Frequency")
        st.pyplot()
    elif choice == "SkinThickness":
        st.subheader("Histogram")
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(111)
        ax.hist(df_copy['SkinThickness'], bins=20)
        ax.set_title("Skin_Thickness")
        ax.set_xlabel("Skin_Thickness")
        ax.set_ylabel("Frequency")
        st.pyplot()
    elif choice == "Insulin":
        st.subheader("Histogram")
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(111)
        ax.hist(df_copy['Insulin'], bins=20)
        ax.set_title("Insulin")
        ax.set_xlabel("Insulin")
        ax.set_ylabel("Frequency")
        st.pyplot()
    elif choice == "BMI":
        st.subheader("Histogram")
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(111)
        ax.hist(df_copy['BMI'], bins=20)
        ax.set_title("BMI")
        ax.set_xlabel("BMI")
        ax.set_ylabel("Frequency")
        st.pyplot()
    elif choice == "DiabetesPedigreeFunction":
        st.subheader("Histogram")
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(111)
        ax.hist(df_copy['DiabetesPedigreeFunction'], bins=20)
        ax.set_title("Diabetes_Pedigree_Function")
        ax.set_xlabel("Diabetes_Pedigree_Function")
        ax.set_ylabel("Frequency")
        st.pyplot()
    elif choice == "Age":
        st.subheader("Histogram")
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(111)
        ax.hist(df_copy['Age'], bins=20)
        ax.set_title("Age")
        ax.set_xlabel("Age")
        ax.set_ylabel("Frequency")
        st.pyplot()
    elif choice == "Has Diabetese":
        st.subheader("Histogram")
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(111)
        ax.hist(df_copy['Has Diabetese'], bins=20)
        ax.set_title("Has_Diabetese")
        ax.set_xlabel("Has_Diabetese")
        ax.set_ylabel("Frequency")
        st.pyplot()
    else:
        st.write("Please select a column")
        
    # plot boxplot

def Predict():
    st.subheader("Predict")
    st.write("Predict")
    # read pickle file 
    model = pd.read_pickle("../model/model_xgboost.pkl")
    # for columns in df get input from user
    # get input from user
    pregnancies = st.slider('Pregnancies', 0, 17, 0)
    glucose = st.slider('Glucose', 0, 199, 0)
    blood_pressure = st.slider('Blood_Pressure', 0, 199, 0)
    skin_thickness = st.slider('Skin_Thickness', 0, 99, 0)
    insulin = st.slider('Insulin', 0, 99, 0)
    bmi = st.slider('BMI', 0, 99, 0)
    diabetes_pedigree_function = st.slider('Diabetes_Pedigree_Function', 0, 99, 0)
    age = st.slider('Age', 0, 99, 0)
    # create dataframe
    data = {'Pregnancies': pregnancies, 'Glucose': glucose, 'Blood_Pressure': blood_pressure, 'Skin_Thickness': skin_thickness, 'Insulin': insulin, 'BMI': bmi, 'Diabetes_Pedigree_Function': diabetes_pedigree_function, 'Age': age}
    df_check = pd.DataFrame(data, index=[0])
    # predict button
    if st.button("Predict"):
        result = model.predict(df_check)
        # if result is 1 then user has diabetes
        if result == 1:
            st.warning("Based on the history the user has a high probablity of having diabetes")
        else:
            st.success("Based on the history the user has a less probablity of having diabetes")



def main():
    st.title("Streamlit Tutorial")
    # sidebar chooserr 
    st.sidebar.header("User Input")
    st.sidebar.subheader("Choose your option")
    # clickable sidebar buttons
    option = st.sidebar.radio("Choose", ("Introduction", "Predict"))
    if option == "Introduction":
        Introduction()
    elif option == "Predict":
        Predict()




main()