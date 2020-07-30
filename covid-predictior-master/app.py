import numpy as np
import pickle
import pandas as pd
import streamlit as st 

from PIL import Image
pickle_in = open("finalized_model.pkl","rb")
classifier=pickle.load(pickle_in)

def welcome():
    return "Welcome All"

def predict_class(Age,Fever,BodyPains,RunnyNose,Difficulty_in_Breath): 
    prediction=classifier.predict([[Age,Fever,BodyPains,RunnyNose,Difficulty_in_Breath]])
    return prediction



def main():
    st.title("covid predictior")
    html_temp = """
    <div style="background-color:blue;padding:10px">
    <h2 style="color:white;text-align:center;">covid prediction using ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    Age = st.slider('Age',0,100)
    Fever = st.slider('Fever',0,120)
    BodyPains = st.text_input('BodyPains 0(no)-1(yes)'," ")
    RunnyNose = st.text_input('RunnyNose 0(no)-1(yes)'," ")
    Difficulty_in_Breath = st.text_input('Difficulty_in_Breath 0(no)-1(yes)'," ")
    result=""
    if st.button("Predict"):
        result=predict_class(int(Age),int(Fever),int(BodyPains),int(RunnyNose),int(Difficulty_in_Breath))
        if result==0:
            result="you dont have covid-19"
        else:
            result="sorry,you have covid-19"
        
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("My Linkedin Link-https://www.linkedin.com/in/achutha-subhash2b29a167/")
        st.text("My Github Link-https://github.com/achuthasubhash")

if __name__=='__main__':
    main()
    
