import os
import numpy as np
import pandas as pd
import flask
import pickle
from flask import Flask, render_template, request


app = Flask(__name__)

@app.route('/', methods = ['GET','POST'])
def index():
    result=""
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        print(to_predict_list)
        result=ValuePredictor(to_predict_list).item()
        result="{0:.0%}".format(result)
        print(result)
        #if int(result) == 1:
         #   prediction = 'Patient has overdose risk.'
        #else:
         #   prediction = 'Patient has no overdose risk.'
        #return render_template("result.html", prediction=prediction)
        #return render_template('index.html', predict = result)
    return render_template('index.html', predict = result)
#def index():
#    return flask.render_template('index.html')



#@app.route('/result',methods = ['POST'])
#def result():
#    if request.method == 'POST':
#        to_predict_list = request.form.to_dict()
#        to_predict_list = list(to_predict_list.values())
#        to_predict_list = list(map(int, to_predict_list))
#        print(to_predict_list)
#        result=ValuePredictor(to_predict_list).item()
#        result="{0:.0%}".format(result)
        #if int(result) == 1:
         #   prediction = 'Patient has overdose risk.'
        #else:
         #   prediction = 'Patient has no overdose risk.'
        #return render_template("result.html", prediction=prediction)
#        return result

@app.route('/result2', methods=['POST'])
def result2():
    if request.method == 'POST':
        ps = pd.read_csv("Dataset/pres_test.csv")
        X = ps.loc[:, ].values
        result=ValuePredictor2(X)
        sum = 0
        for i in result:
            sum = sum + result[i]
    return render_template("result2.html", prediction=sum, pred=len(result))

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 7)
    loaded_model = pickle.load(open("model.pkl", "rb"))
    #result = loaded_model.predict(to_predict)
    result = loaded_model.predict_proba(to_predict)
    #return result[0]
    return result[:, 1]

def ValuePredictor2(X):
    loaded_model = pickle.load(open("model2.pkl", "rb"))
    result = loaded_model.predict(X)
    return result


if __name__ == "__main__":
    app.run(debug=True, port=8090)