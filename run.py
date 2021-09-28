from flask import Flask, render_template, url_for, redirect
from flask.templating import render_template_string
from sklearn import svm
from svm_model import SvmModel
from knn_model import KnnModel
from logistic_model import LogisticModel
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    logistic_result = LogisticModel.logistic()
    return render_template('index.html', result = logistic_result)


@app.route('/svm_result')
def svm_result():
    svm_result = SvmModel.svm()
    return render_template('svm.html', result = svm_result)


@app.route('/knn_result')
def knn_result():
    knn_result = KnnModel.knn()
    return render_template('knn.html', result = knn_result)


if __name__ == '__main__':
    app.run(debug=True)
© 2021 GitHub, Inc.
Terms
