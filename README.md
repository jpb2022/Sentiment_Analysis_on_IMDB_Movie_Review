# Sentiment Analysis of IMDb Movie Reviews

## Project Overview
This project focuses on sentiment analysis of IMDb movie reviews, classifying them as either positive or negative. The analysis involves text preprocessing, feature extraction, model training, and evaluation. Various machine learning models, including Logistic Regression, Support Vector Machines (SVM), and Multinomial Naive Bayes, are used to determine sentiment polarity.

## Dataset
- **Source:** IMDb dataset (CSV format)
- **Content:** Movie reviews and their corresponding sentiment labels (positive or negative)

## Technologies Used
- Python
- NumPy, Pandas (Data Manipulation)
- Seaborn, Matplotlib (Data Visualization)
- NLTK (Natural Language Processing)
- BeautifulSoup (Text Cleaning)
- Scikit-learn (Machine Learning Models & Evaluation)

## Project Workflow

### 1. Data Loading
The dataset is loaded into a Pandas DataFrame for further processing.

### 2. Data Exploration
- Checking dataset shape, structure, and distribution of sentiment labels.
- Displaying basic statistics and sample reviews.

### 3. Data Preprocessing
- **Text Cleaning:** Removing HTML tags, special characters, and noisy text.
- **Tokenization:** Splitting text into words.
- **Stopword Removal:** Removing commonly used words that do not add value to sentiment analysis.
- **Text Normalization:** Stemming words to their root form.

### 4. Text Vectorization
- **Bag of Words (BoW):** Using `CountVectorizer` to convert text data into numerical features.
- **TF-IDF (Term Frequency-Inverse Document Frequency):** Using `TfidfVectorizer` for feature extraction.

### 5. Model Training
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Multinomial Naive Bayes**

### 6. Model Evaluation
- Predicting sentiment on the test dataset.
- Evaluating models using:
  - **Accuracy Score**
  - **Classification Report (Precision, Recall, F1-Score)**
  - **Confusion Matrix**
- Comparing BoW and TF-IDF feature extraction methods.

### 7. Data Visualization
- **Confusion Matrices** for model performance assessment.
- **Word Clouds** for visualizing frequently occurring words in positive and negative reviews.

## Key Findings
- **TF-IDF vectorization** performed better than BoW in sentiment classification.
- **Logistic Regression** achieved the highest accuracy among all models.
- Word cloud visualizations provided insights into commonly used words in positive and negative reviews.

## Installation & Usage
### Prerequisites
Ensure you have Python and the necessary libraries installed:
```sh
pip install numpy pandas seaborn matplotlib nltk beautifulsoup4 scikit-learn textblob wordcloud
```

### Running the Project
1. Clone this repository:
```sh
git clone https://github.com/yourusername/sentiment-analysis-imdb.git
cd sentiment-analysis-imdb
```
2. Run the Jupyter Notebook:
```sh
jupyter notebook
```
3. Execute the notebook cells to see the results.

## Conclusion
This project demonstrates the application of **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques in sentiment analysis. The results highlight the effectiveness of **TF-IDF vectorization** and **Logistic Regression** in classifying IMDb movie reviews. Future improvements may include deep learning models for enhanced accuracy.

## License
This project is open-source and available under the MIT License.

## Contact
For any queries, feel free to reach out:
- **Email:** your-email@example.com
- **GitHub:** [jpb2022](https://github.com/jpb2022)

