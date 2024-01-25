import tensorflow as tf
import numpy as numpy
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import keras
from distutils.dir_util import copy_tree, remove_tree
from PIL import Image
from random import randint
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.metrics import balanced_accuracy_score as BAS
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow_addons as tfa
from keras.utils.vis_utils import plot_model
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG
from tensorflow.keras.layers import SeparableConv2D, BatchNormalization, MaxPool2D
from flask import *
from os import listdir
from os.path import isfile, join


app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")


model = keras.models.load_model(
    'C:/Users/Abhishek Walinjkar/Desktop/AD_CNN/models/alzheimer_cnn_model')
print("+"*50, "Model is loaded")


@app.route("/prediction", methods=["POST"])
def prediction():
    try:
        f1 = request.files['file1']
        if f1 == '':
            f1 = None
            prediction = "Image not provided!"

    except:
        f1 = None
        prediction = "No data provided for MRI!"

    if f1 is not None:
        IMAGE_SIZE = [224, 224]
        b = 0

        f1.save(f1.filename)
        name = f1.filename
        img = name

        img = tf.io.read_file(img)
        # convert the string to a uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # convert to floats in the [0,1] range
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size (224,224).
        img = tf.image.resize(img, IMAGE_SIZE)

        # img=np.reshape(img,(1,180,180,3))
        img = numpy.reshape(img(1, 224, 224, 3))

        b = model.predict(img)
        b = numpy.round(b)

        absolute_val_array = numpy.abs(b - 1)
        smallest_difference_index = absolute_val_array.argmin()

        diseases = {0: "Non Demented", 1: "Very Mildly Demented",
                    2: "Mildly Demented", 3: "Moderately Demented"}
        print('Person might have', diseases[smallest_difference_index])
        ab = diseases[smallest_difference_index]
        prediction = 'Person might be : {}'.format(ab)

        os.remove(f1.filename)

    return render_template("predict.html", data=prediction)


if __name__ == "__main__":
    app.run(debug=True)
