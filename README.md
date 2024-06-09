# Table-Classification-from-Financial-Statements

## Introduction
The objective of this project is to classify tables extracted from financial statements into five distinct categories. This task involves several steps, including data extraction, preprocessing, model training, and model selection.

## Workflow

1. **Data Extraction**
   - Utilized Beautiful Soup, a Python library for parsing HTML documents, to extract text data from tables within financial statements.
   - Structured the extracted text data into DataFrames, assigning each entry to its respective class for further processing and analysis.

2. **Data Cleaning and Processing**
   - Identified and removed missing values within the dataset to ensure completeness and reliability of the data.
   - Transformed all text data into lowercase to maintain consistency, as NLP techniques are case-sensitive. This step helps in reducing the dimensionality and improving the accuracy of the model.
   - Applied lemmatization to convert words to their base or root form, which helps in standardizing the data and improving the performance of the NLP models.

3. **Feature Engineering**
   - Applied the Term Frequency-Inverse Document Frequency (TF-IDF) technique to convert textual data into numerical vectors.
   - TF-IDF represents the importance of each word in the documents relative to the entire dataset.

4. **Model Development**
   - Developed six different Machine learning classification models.
   - Each model was trained on the pre-processed dataset to classify tables from financial statements into distinct categories.

5. **Model Evaluation**
   - Evaluated the performance of each model using accuracy and f1-score as metrics to assess their classification accuracy on the test dataset.

6. **Model Selection**
   - Out of six models, SVM and Random Forest classifier performed well. Further, utilized cross-validation techniques to validate the robustness of the models across different subsets of the data.
   - After thorough evaluation and cross-validation, the Random Forest classifier emerged as the top-performing model.

## Conclusion
Random Forest consistently demonstrated superior performance in Accuracy across all subsets of the data, leading to its selection as the final model.

## Web Application
- I developed a Streamlit web application as an additional enhancement to the project, providing further functionality and accessibility.
- Users can upload financial statements and classify tables using the trained Random Forest model.

- ![image](https://github.com/JayaKrishanS/Table-Classification-from-Financial-Statements/assets/129932233/aba6000e-0b64-40eb-a5b0-954bbb083a25)


## Links
- [App link](https://jk-table-classification-from-financial-statements.streamlit.app/)
- [GitHub Repository](https://github.com/JayaKrishanS/Table-Classification-from-Financial-Statements.git)

## Note
- To test the model's performance, you can execute the 'test_model.py' file on your system. It's essential to note that the 'model.pkl' and 'vectorizer.pkl' files are required for the functionality of the model.
- Alternatively, you can directly access the model through the web application, the link to which is mentioned above.
