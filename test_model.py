#Importing libraries
import pickle
import pandas as pd
import re
from bs4 import BeautifulSoup
import sys
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet', quiet=True)


# Importing the model and TF-IDF file
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

def extract_words_from_html(file_path):
    """Extract words from a single HTML file, excluding numbers."""
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

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

    #Vectorizing the input
    input_features = vectorizer.transform([lemmatized_words])
    input_prediction = model.predict(input_features)

    if input_prediction[0] == 0:
        print("Balance Sheets")
    elif input_prediction[0] == 1:
        print("Cash Flow")
    elif input_prediction[0] == 2:
        print("Income Statement")
    elif input_prediction[0] == 3:
        print("Notes")
    else:
        print("Others")

if __name__ == "__main__":
    if len(sys.argv) == 2:
        file_path = sys.argv[1]
    else:
        file_path = input("Enter the HTML file path: ")
        prediction(file_path)