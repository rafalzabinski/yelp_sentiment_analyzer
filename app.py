from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
import pickle
import sklearn
import numpy as np 
import pandas as pd 
import nltk 
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string
from nltk.corpus import stopwords 
from flask import send_file
#nltk.download('stopwords')
ENGLISH_STOP_WORDS = stopwords.words('english')


def my_tokenizer(sentence):

    listofwords = sentence.strip().split()         
    listof_words = []    
    for word in listofwords:
        if not word in ENGLISH_STOP_WORDS:
            lemm_word = WordNetLemmatizer().lemmatize(word)
            for punctuation_mark in string.punctuation:
                word = word.replace(punctuation_mark, '').lower()
            if len(word)>0:
                listof_words.append(word)
    return(listof_words)


import re 
def function_clean(text):
    text = re.sub(r"http\S+", "", text) 
    text = re.sub("@[^\s]*", "", text)
    text = re.sub("#[^\s]*", "", text)
    text = re.sub('[0-9]*[+-:]*[0-9]+', '', text)
    text = re.sub("'s", "", text)   
    return text

#tfidf = pickle.load(open("tfidf.pickle", 'rb'))
#lr_model = pickle.load(open("Regressor_model.sav", 'rb'))
tfidf = pickle.load(open("100k_vectorizer.pickle", 'rb'))
lr_model = pickle.load(open("100k_regression.sav", 'rb'))
rfc_model = pickle.load(open("rf_model.sav", 'rb'))


app = Flask(__name__)


@app.route("/")
def index():
    data = {"input": "", "result": ""}
    return render_template("index.html", data=data, methods=["GET", "POST"])
    
    


@app.route('/submit', methods=['POST'])
def submit():
    user_input = request.form['text']
    result = lr_model.predict(tfidf.transform([function_clean(user_input)]))
    result2 = rfc_model.predict(tfidf.transform([function_clean(user_input)]))
    #return render_template("index.html", data= {"input": request.form['text'], "result": result})
    if result == 1:
       filename = 'static/positive.png'
       sentiment = 'Positive!'
    else:
       filename = 'static/negative.png'
       sentiment = 'Negative!'

    if result2 == 1:
       filename2 = 'static/positive.png'
       sentiment2 = 'Positive!'
    else:
       filename2 = 'static/negative.png'
       sentiment2 = 'Negative!'

    if user_input == 'person woman man camera tv':
       filename2 = 'static/trump.png'
       sentiment2 = 'NOT cognitively there!'
       filename = 'static/trump.png'
       sentiment = 'NOT cognitively there!'


    #return send_file(filename, mimetype='image/gif')
    #return render_template("index.html", "result":filename})
    #return render_template("index.html", image = filename, data= {"input": request.form['text'], "result": sentiment})
    return render_template("index.html", data= {"image_lr":filename, "image_rfc":filename2, "input": request.form['text'], "result_lr": sentiment, "result_rfc": sentiment2})



    


if __name__ == '__main__':
    app.run(debug=True)
