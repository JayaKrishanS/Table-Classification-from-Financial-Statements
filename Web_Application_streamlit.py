import pickle
import pandas as pd
import re
from bs4 import BeautifulSoup
import nltk
from nltk.stem import WordNetLemmatizer
import streamlit as st
import pybase64
from io import StringIO

nltk.download('wordnet', quiet=True)

# Importing the model and TF-IDF file
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))



def extract_words_from_html(uploaded_file):
    """Extract words from an uploaded HTML file, excluding numbers."""
    # Read the uploaded HTML file
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    html_content = stringio.read()
    
    # Parse the HTML content and extract all text
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text()

    # Clean the extracted text to keep only words and exclude numbers
    words_list = re.findall(r'\b[a-zA-Z]+\b', text)
    words = ' '.join(words_list)
    return words


def prediction(file_path):
    words = extract_words_from_html(file_path)

    #Changing into lower case
    words = words.lower()
    #removing words less than three letters
    words_list = words.split()
    filtered_words = [word for word in words_list if len(word) > 3]
    words= ' '.join(filtered_words)

    #Applying lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = ' '.join([lemmatizer.lemmatize(word) for word in words.split()])

    #Vectorizing the input data
    input_features = vectorizer.transform([lemmatized_words])
    input_prediction = model.predict(input_features)

    if input_prediction[0] == 0:
        result = "This file belongs to 'Balance Sheets' "
    elif input_prediction[0] == 1:
        result =  "This file belongs to 'Cash Flow' "
    elif input_prediction[0] == 2:
        result = "This file belongs to 'Income Statement' "
    elif input_prediction[0] == 3:
        result = "This file belongs to 'Notes' "
    else:
        result = "This file belongs to 'Others' "
    
    return result


def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return pybase64.b64encode(data).decode()

img = get_img_as_base64("web_app_background.jpg")
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/png;base64,{img}");
background-size: 100%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown("<h2 style='color: orange;'> Table Classification from Financial Statements </h2>", unsafe_allow_html = True)

tab1,tab2,tab3,tab4 = st.tabs(["Predict"," "," ", "About project"])

with tab4:
    st.markdown("<h5 style='color: orange;'>Summary :</h5>", unsafe_allow_html=True) 
    st.markdown("* The objective of this project is to classify tables in financial statements into one of five distinct classes. (Balance Sheets, Cash Flow, Income Statement, Notes, Others)")
    st.markdown("* Words from tables in financial statements were scraped using Beautiful Soup and created a dataframe with corresponding classes.")
    st.markdown("* Handled missing data and performed various data preprocessing steps to clean the dataset")
    st.markdown("* Employed the TF-IDF (Term Frequency-Inverse Document Frequency) technique to convert textual data into numerical vectors suitable for machine learning models.")
    st.markdown("* Developed and tested six different machine learning models. Out of that Support Vector Machine (SVM) and Random Forest classifiers achieved an accuracy of 94%.")
    st.markdown("* After conducting cross-validation, the Random Forest classifier was selected for its superior performance across different datasets.")

with tab1:
    uploaded_file = st.file_uploader("Choose an HTML file", type="html")
    if st.button('Predict'):
        if uploaded_file is not None:
            result = prediction(uploaded_file)
            st.success(result)
        else:
            st.error("Please upload a HTML file for prediction")
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")
    st.markdown("Made by: Jaya Krishna S")