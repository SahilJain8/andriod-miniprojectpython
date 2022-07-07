from flask import Flask,request,jsonify,render_template
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import pickle
import tensorflow
from tensorflow import keras
import os
from tensorflow.keras.preprocessing.image import load_img
model = keras.models.load_model('Mymodel.h5')
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   fill_mode='nearest')
training_set = train_datagen.flow_from_directory("..\\Dataset\\New Plant Diseases Dataset(Augmented)\\New Plant Diseases Dataset(Augmented)\\train",
                                                 target_size=(224, 224),
                                                 batch_size=128,
                                                 class_mode='categorical')

class_dict = training_set.class_indices
li = list(class_dict.keys())
app = Flask(__name__)






def load_image(img_path):

    new_img = keras.utils.load_img(img_path, target_size=(224, 224))
    img = keras.utils.img_to_array(new_img)
    img = np.expand_dims(img, axis=0)
    img = img/255                               

    return img

def prediction(img_path):
    new_image = load_image(img_path)
    
    prediction = model.predict(new_image)
    d = prediction.flatten()
    j = d.max()
    for index,item in enumerate(d):
        if item == j:
            class_name = li[index]
    print(class_name)
    return class_name


@app.route("/", methods=['GET', 'POST'])
def home():

    return render_template('home.html')

@app.route("/predict", methods = ['GET','POST'])
def predict():
    
    if request.method == 'POST':
        
        file = request.files['file']
        filename = file.filename
        file_path = os.path.join(r'C:/Users/jains/Desktop/Projects/Machine learning/MYapp/static/uploads', filename)                       #slashes should be handeled properly
        file.save(file_path)
        print(filename)
        product = prediction(file_path)
       
        
    return jsonify({'The condition is':str(product)})


if __name__ == "__main__":
    app.run()