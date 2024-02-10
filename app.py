import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__, template_folder='/content/templates', static_folder='/content/static')
model = pickle.load(open('/content/model.pkl','rb'))

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/predict",methods = ["POST"])
def predict():
    Delivery_person_Age = int(request.form["Delivery_person_Age"])
    Delivery_person_Ratings = float(request.form["Delivery_person_Ratings"])
    distance = int(request.form["distance"])
    features = np.array([[Delivery_person_Age,Delivery_person_Ratings,distance]])

    prediction = model.predict(features)
    output = np.round(prediction[0], 2)
    return render_template('index.html', prediction_text='Predicted Delivery Time in Minutes = {}'.format(output))


if __name__=="__main__":
   app.run()