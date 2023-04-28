import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template

#creating app
app = Flask(__name__)

# laoding model
model = pickle.load(open('model.pkl','rb'))
train = pd.read_csv('X_train.csv')
#giving paths
@app.route('/')
def index():
    # sending by default data
    Age  = train['Age'].values
    Height  = train['Height'].values
    Weight  = train["Weight"].values
    Duration  = train['Duration'].values
    Heart_Rate  = train['Heart_Rate'].values
    Body_Temp  = train['Body_Temp'].values
    print(Age)
    return render_template('index.html',a=Age,h=Height,w=Weight,d=Duration,hr=Heart_Rate,b=Body_Temp)

@app.route('/predict',methods=['POST'])
def predict():
    Gender  = request.form['Gender']
    Age = request.form['Age']
    Height = request.form['Height']
    Weight = request.form['Weight']
    Duration = request.form['Duration']
    Heart_Rate = request.form['Heart_Rate']
    Body_Temp = request.form['Body_Temp']

    featurs = np.array([[Gender,Age,Height,Weight,Duration,Heart_Rate,Body_Temp]])
    pred = model.predict(featurs).reshape(1,-1)
    return render_template('index.html', output = pred[0])

#python main
if __name__ == "__main__":
    app.run(debug=True)