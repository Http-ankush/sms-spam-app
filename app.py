import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    # Lowercase and tokenize
    text = text.lower()
    words = nltk.word_tokenize(text)

    # Remove non-alphanumeric
    words = [word for word in words if word.isalnum()]

    # Remove stopwords and punctuation
    words = [word for word in words if word not in stopwords.words('english') and word not in string.punctuation]

    # Stemming
    words = [ps.stem(word) for word in words]

    return " ".join(words)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("sms_spam_classifier")

input = st.text_area("enter the message")

if st.button('predict'):
    transformed_sms = transform_text(input)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
