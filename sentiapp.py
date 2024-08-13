from flask import Flask, make_response, request, render_template
import io
from io import StringIO
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)
lm = WordNetLemmatizer()
cv = CountVectorizer(ngram_range=(1,2))
def text_transformation(df_col):
    corpus = []
    for item in df_col:
        new_item = re.sub('[^a-zA-Z]',' ',str(item))
        new_item = new_item.lower()
        new_item = new_item.split()
        new_item = [lm.lemmatize(word) for word in new_item if word not in set(stopwords.words('english'))]
        corpus.append(' '.join(str(x) for x in new_item))
    return corpus

@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')


@app.route('/predict', methods=["POST"])
def predict():
    f = request.files['data_file']
    if not f:
        return render_template('index.html', prediction_text="No file selected")

    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
    result = stream.read()#.replace("=", ",")
    print(result)
    df = pd.read_csv(StringIO(result),names=['text'])
    print(df.head())
    df.columns = ['text']
    df_corpus = text_transformation(df['text'])
    print(df_corpus)
    cv = pickle.load(open("cv.pkl", 'rb'))

    data = cv.transform(df_corpus)
    
    #Feature Engineering
    #df = feature_engineering(df)
    
    #X = scalar(df)
    
    # load the model from disk
    loaded_model = pickle.load(open("lg_model.pkl", 'rb'))
    
    print (loaded_model)

    result = loaded_model.predict(data)
    
    return render_template('index.html', prediction_text="Predicted Salary is/are: {}".format(result))

if __name__ == "__main__":
    app.run(debug=False,port=5000)