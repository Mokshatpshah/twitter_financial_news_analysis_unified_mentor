# Financial Tweet Sentiment Classifier

This is an end-to-end machine learning project that classifies financial tweets based on sentiment (e.g., Positive, Negative, Neutral). It covers everything from data cleaning, visualization, model training, and evaluation to a real-time prediction interface using Streamlit.

---

## Project Structure

```
.
├── dashboard1.py           # Streamlit app for real-time sentiment prediction
├── financial_tweets.xlsx   # Cleaned dataset with tweet text and sentiment labels
├── model.plk               # Trained Logistic Regression model
├── vectorizer.plk          # TF-IDF vectorizer used for transforming tweet text
├── train_data.xlsx         # Original dataset for training
├── train_model.ipynb       # Jupyter Notebook for data preprocessing, training, and evaluation
|- demo video
```

---

## Features

- Cleans and preprocesses tweet text (removes URLs, mentions, punctuation, digits, etc.)
- Visualizes tweet sentiment distribution and generates a WordCloud
- Trains a Logistic Regression classifier using TF-IDF features
- Evaluates model performance using classification report and confusion matrix
- Interactive Streamlit app for classifying financial tweets in real-time

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/financial-tweet-classifier.git
cd financial-tweet-classifier
```

### 2. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` yet, create one with:

```bash
pip freeze > requirements.txt
```

Or install manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn wordcloud joblib streamlit
```

---

## Model Training (Optional)

You can retrain the model using the provided notebook:

- Open `train_model.ipynb` in Jupyter
- It will:
  - Load and clean data
  - Visualize distributions and WordCloud
  - Vectorize text using TF-IDF
  - Train a Logistic Regression model
  - Save the model and vectorizer (`model.plk`, `vectorizer.plk`)

---

## Running the Streamlit App

Ensure `model.plk` and `vectorizer.plk` are present in the same directory as `dashboard1.py`.

To launch the app:

```bash
streamlit run dashboard1.py
```

Once it starts, you'll see a browser window where you can enter a financial tweet and receive the predicted sentiment instantly.

---

