# NLP_DISASTER_TWEETS
## NLP Disaster Tweets Classification

This repository contains the code for the Kaggle "Disaster Tweets" competition. The objective is to build a machine learning model that can classify tweets as either a real disaster (`target=1`) or not a disaster (`target=0`).

### 🚀 Project Overview

The project follows a standard machine learning workflow, with a particular focus on natural language processing techniques for text data.

1.  **Data Loading & Cleaning**: The raw `train.csv` and `test.csv` files are loaded into pandas DataFrames, and an initial check for missing values and duplicates is performed.
2.  **Exploratory Data Analysis (EDA)**: Initial data insights are gathered through visualizations, including a pie chart to check the balance of the target variable and word clouds to identify the most frequent keywords in both disaster and non-disaster tweets.
3.  **Text Preprocessing & Feature Engineering**: A robust pipeline is created to clean and transform the tweet text. This includes:
    * **Text Cleaning**: A `preprocessing_complete` function is used to handle lowercasing, URLs, HTML tags, and other noise.
    * **Normalization**: Words are normalized using lemmatization from spaCy, and a custom `remove_repeated` function handles elongated words (e.g., "sooo" becomes "so").
    * **Stopword Removal**: A `rm_stopwords` function is used to remove common words that don't add semantic value.
    * **Numerical Features**: Handcrafted features like `sent_count`, `word_col`, `char_count`, `hashtag`, and `avg_words` are created to provide additional information to the model.
    * **TF-IDF Vectorization**: The cleaned text is converted into numerical features using `TfidfVectorizer`.
4.  **Model Training & Evaluation**: Several classifiers are trained and evaluated on a validation set to compare their performance. The key evaluation metric for this competition is the **F1-score**. The performance of different models is visualized to facilitate a clear comparison.
5.  **Submission**: The final model (Logistic Regression in this case) is trained on the complete training data and used to predict the `target` for the test set, creating a submission file in the required format.

### 🛠️ Technology Stack

* **Python**
* **pandas**: For data manipulation and analysis.
* **numpy**: For numerical operations.
* **scikit-learn**: For machine learning models, data splitting, and performance metrics.
* **nltk & spaCy**: For natural language processing, including tokenization and lemmatization.
* **matplotlib & seaborn**: For data visualization.
* **wordcloud**: For generating word cloud visualizations.

### 🔮 Future Work

* **Advanced Models**: Explore deep learning models such as Recurrent Neural Networks (RNNs) or Transformer-based models (e.g., BERT) for potentially higher accuracy.
* **Hyperparameter Tuning**: Use techniques like `GridSearchCV` or `RandomizedSearchCV` to fine-tune the hyperparameters of the best-performing model to optimize its F1-score.
* **Additional Feature Engineering**: Experiment with new features like sentiment scores, the number of exclamations, or the presence of specific keywords to further improve model performance.
