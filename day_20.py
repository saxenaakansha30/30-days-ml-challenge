# Problem: Create a topic model using Latent Dirichlet Allocation (LDA)
# Dataset: https://www.kaggle.com/datasets/aashita/nyt-comments?resource=download (ArticlesApril2017.csv)

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim import corpora
import gensim
import pyLDAvis.gensim


# Step 1: Load the data
data = pd.read_csv('dataset/ArticlesApril2017.csv')
# print(data.head())
# print(data.info())

# Select the snippet column
# We use the snippet column, which contains short text snippets of the articles.
# This will be our source of text for topic modeling.
text_data = data['snippet'].dropna().tolist()

# Step 2: Preprocessing Setup
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# Step 3: Text Preprocessing
def preprocess(text):
    # Convert to lowercase
    text = text.lower()
    # Remove unwanted punctuations and numbers
    text = re.sub(r'\W+', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stop words and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens


# Step 4: Apply preprocessing to text data
cleaned_data = [preprocess(text) for text in text_data]

# Step 5: Create a dictionary and corpus of LDA
dictionary = corpora.Dictionary(cleaned_data)
corpus = [dictionary.doc2bow(text) for text in cleaned_data]

# Step 6: Build the LDA Model
lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)

# Step 7: Display topics
topics = lda_model.print_topics(num_topics=10)
for topic in topics:
    print(topic)

# Step 8: Visualize topics using pyLDAvis
# pyLDAvis.enable_notebook()
# lda_vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
# pyLDAvis.display((lda_vis))