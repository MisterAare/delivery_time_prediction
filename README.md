# Food delivery time prediction



# Deployment:
The model was deployed og Google Colab IDE using the Flask framework and HTML for the user interface.

```python
from google.colab.output import eval_js
print(eval_js("google.colab.kernel.proxyPort(5000)"))
```

```python
* https://3ptu6ergm1w-496ff2e9c6d22116-5000-colab.googleusercontent.com/
```

```python
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
```

HERE'S WHAT THE FRONT-END LOOKS LIKE:

![chrome_Pe0v1wp06x](https://github.com/MisterAare/delivery_time_prediction/assets/109184556/2c3cab6a-bd63-455a-823f-7bae2fb7c084)
