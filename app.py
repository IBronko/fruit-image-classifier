from pyparsing import col
import streamlit as st
from fastai.vision.all import *
from fastai.vision.widgets import *
import json
from streamlit_lottie import st_lottie

#####################################
# Streamlit page configuration
#####################################

st.set_page_config(
     page_title="Thai-Fruit Classifier",
     page_icon="🍌",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': 'https://github.com/IBronko/',
         'Report a bug': "https://github.com/IBronko/",
         'About': "# This is a personal project."
     }
 )

#####################################
# Display lottie file 
#####################################

st.markdown("<h1 style='text-align: center;'>Welcome, I am a Thai-Fruit classifier.</h1>", unsafe_allow_html=True)

def load_lottiefile(filepath: str):
        with open(filepath, "r") as f:
            return json.load(f)
        
lottie_coding = load_lottiefile("lottie.json")
col1, col2, col3 = st.columns(3)
with col2:   
    st_lottie(
        lottie_coding,
        speed=1,
        reverse=False,
        loop=True,
        quality="medium", # medium ; high,
        key=None,
        )


fruit_dict = {
    'custard apple':"This looks like a Custard Apple to me.", 
    'dragon fruit':"This looks like a Dragon fuit to me.", 
    'durian':"This looks like a Durian to me.", 
    'jackfruit':"This looks like a Jack Fruit to me.", 
    'mangosteen':"This looks like a Mangosteen to me.", 
    'nam dok mai mango':"This looks like a Nam Dok Mai Mango to me.", 
    'pomelo':"This looks like a Pomelo to me.", 
    'rambutan':"This looks like a Rambutan to me.", 
    'rose apple':"This looks like a Rose Apple to me.", 
    'salak fruit':"This looks like a Salak Fruit to me.", 
    'sapodilla':"This looks like a Sapodilla to me.", 
    'tamarind':"This looks like a Tamarind to me.", 
    'thai bananas':"This looks like a Thai Banana to me."
    }


uploaded_image = st.file_uploader("Upload your image and I'll give it a try.", type=["png", "jpg"])
if uploaded_image is not None:
    
    st.image(uploaded_image)
    
    with st.spinner("Loading your classifier..."):
        model = load_learner("fruit_classifier.pkl")
    
    try:
        pred,pred_idx,probs = model.predict(uploaded_image.getvalue())
        st.success(f"{fruit_dict[pred]} I am {probs[pred_idx]*100:.0f}% confident.")
        st.caption(f"Caution: I have only been trained on a small set of images. I may also be wrong.")
    except:
        st.write("Sorry, I don't know that fruit")    
    
with st.expander("Info"):
     st.write("""
         Some Text.
     """)