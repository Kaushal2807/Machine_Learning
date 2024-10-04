import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

# Create a flask app
app = Flask(__name__)

# Load the pickle model
model = pickle.load(open("model1.pkl","rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)

    return render_template("index.html",prediction_text="The flower spieces is {}".format(prediction[0]))

if __name__ == "__main__":  
    app.run(debug=True)