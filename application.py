import pickle
from flask import Flask,request,render_template
import numpy as np
import pandas as pd


application = Flask(__name__)
app = application


## importing the linear regressiona and scaler pickle...
linear_regression = pickle.load(open('models/linearRegression.pkl','rb'))
standard_scaler = pickle.load(open('models/scaler.pkl','rb'))


## Route for home page...
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        Temperature=float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        ## Remember only .transform() on test data...
        scaled_input = standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])

        ## Predicting...
        result = linear_regression.predict(scaled_input)

        return render_template('home.html',result=result[0])

    else:
        return render_template('home.html')
    


if __name__ == "__main__":
    app.run(debug=True)
