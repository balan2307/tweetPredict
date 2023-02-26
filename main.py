import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from nltk.corpus import stopwords
from sklearn.metrics import roc_auc_score

nltk.download('stopwords')
from flask import Flask, render_template, url_for, request
import pandas as pd
from sklearn import preprocessing
import numpy as np

df = pd.read_csv('project.csv')

le = preprocessing.LabelEncoder()
df['Type'] = le.fit_transform(df['Type'])

stopset = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset)

y = df.Type

x = vectorizer.fit_transform(df.Tweet)

X_train, X_test, Y_train, Y_test = train_test_split(x, y, random_state=42)

model = naive_bayes.MultinomialNB()
model.fit(X_train, Y_train)

# load the model from disk
app = Flask(__name__, template_folder='templates')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        inp = request.form['message']
        emergency = np.array([inp])
        emergency_vector = vectorizer.transform(emergency)
        answer = model.predict(emergency_vector)
        # if (answer == 0):
        #     result = "Not an Emergency"
        # else:
        #     result = "Emergency"

        return render_template('result.html', prediction=answer)


if __name__ == '__main__':
    app.run(debug=True)
