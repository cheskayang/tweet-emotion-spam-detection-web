from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import os
from feature_extraction import processData

dest = os.path.join('pickled_classifier', 'pkl_objects')

clf = pickle.load(open(os.path.join(dest, 'classifier.pkl'), 'rb'))
vect = pickle.load(open(os.path.join(dest, 'vect.pkl'), 'rb'))
label = {"0":'non-spam', "1":'spam'}

app = Flask(__name__)

def toPredict(tweet):
    processedTweet = processData(tweet)
    vectorizedTweet = vect.transform(processedTweet)
    result = label[clf.predict(vectorizedTweet)[-1]]
    return result

class TweetForm(Form):
    submittedTweet = TextAreaField('',[validators.DataRequired()])

@app.route('/')
def index():
    form = TweetForm(request.form)
    return render_template('index.html', form=form)

@app.route('/result')
def redirect():
    form = TweetForm(request.form)
    return render_template('index.html', form=form)

@app.route('/result', methods=['POST'])
def predict():
    form = TweetForm(request.form)
    if request.method == 'POST' and form.validate():
        tweet = [request.form['submittedTweet']]
        predictedResult = toPredict(tweet)
        return render_template('index.html', form=form, result="This tweet is " + predictedResult +"!")

    return render_template('index.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)