import tensorflow
from keras.models import load_model
from keras.preprocessing import image
from flask import Flask, render_template, request
import os
import numpy as np

app = Flask(__name__)
model = load_model(r"VegetableImageClassification.h5",compile = False)


@app.route('/')
def index():
    return render_template("input.html")


@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', f.filename)
        f.save(filepath)

        img = image.load_img(filepath,target_size =(150,150))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        pred = np.argmax(model.predict(x),axis=1)
        index = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']
        text = "The classified vegetable is : " +str(index[pred[0]])
    return text


if __name__ == '__main__':
    app.run(debug=True)