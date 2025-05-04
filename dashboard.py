# %%
import pandas as pd
import re

# %%
df = pd.read_csv('train_data.csv')

# %%
df.to_csv('financial_tweets.csv' , index = False)

# %%
df

# %%
def clean_text(text):
 text = re.sub(r'http\S+| www\S+|http\S+' , '' , text, flags = re.MULTILINE)
 text = re.sub(r' \@\w+|\#' , '' , text)
 text = re.sub(r'[^\w\s]' , '' , text) 
 text = re.sub(r' \ d+' , '' , text)
 text = text.lower()
 return text

# %%
df['clean_text'] = df['text'].apply(clean_text)
print(df.head(20))

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# %%
sns.countplot(data = df , x = 'label' , palette = 'viridis')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show

# %%

# %%
from wordcloud import WordCloud
text = ' '. join(df['clean_text'])
wordcloud = WordCloud(width = 800 , height = 400 , background_color = 'white').generate(text)

# %%
plt.figure(figsize= (10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text']).toarray()
y = df['label']


# %%
from sklearn.model_selection import train_test_split

# %%
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix

# %%
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test , y_pred))

# %%
from sklearn.metrics import mean_squared_error
import numpy as np
mse = mean_squared_error(y_test , y_pred)
rmse = np.sqrt(mse)
print("RMSE:" , rmse)

# %%
import seaborn as sns
cm = confusion_matrix(y_test , y_pred)
sns.heatmap(cm , annot=True, fmt='d' , cmap='Blues' , xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# %%
import joblib
joblib.dump(model, 'model.plk')

# %%
import streamlit as st
import joblib
import numpy as np

model = joblib.load('model.plk')

st.title("Financial Tweet Sentiment Classifier")
user_input = st.text_input("enter a financial tweet :")


if st.button("Predict"):
    clean_input = clean_text(user_input)
    input_vector = vectorizer.transform([clean_input]).toarray()
    prediction = model.predict(input_vector)
    st.write(f"Predicted Sentiment: {prediction[0]}")
    
joblib.dump(vectorizer, 'vectorizer.plk')




