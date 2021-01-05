from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Flask utils
from models.model import predict,load_model
from flask import Flask, redirect,url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from flask_ngrok import run_with_ngrok
# Define a flask app

import os
if not os.path.exists('uploads'):
    os.makedirs('uploads')
app = Flask(__name__)
run_with_ngrok(app)

# Model saved with Keras model.save()
MODEL_PATH = 'models/model.h5'

# Load your trained model
melanoma = load_model(MODEL_PATH)
melanoma._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        print(request.form)
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        gender = request.form['gender']
        gender = 0 if gender=="male" else 1
        age = request.form['Age']
        site = request.form['Site']
        result = predict(file_path, melanoma,int(age),int(site),gender)
        return render_template('base.html', prediction_text=result)
    
    return None


if __name__ == '__main__':
    app.run()
