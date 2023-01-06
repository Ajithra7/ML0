import numpy as np
from PIL import Image
import tensorflow as tf

from tensorflow.keras import layers
from model import vgg19_modified as vgg19_ext

IMG_SIZE = (224, 224)
CHANNELS = 3
IMG_SHAPE = IMG_SIZE+(CHANNELS,)
BATCH_SIZE = 32
CLASS_NAMES = ('Apple Pie', 'Biryani', 'Chicken Curry', 'Cupcake', 'Eggs Denedict')
WEIGHT_FILE = "app/weights/best_model_vgg19_2_90val.hdf5"

def classifier(image):
    '''
    Does food classification.
    image : PIL image
    Returns the top prediction with confidence value.
    '''

    # Get the model and load the weights
    model = vgg19_ext(IMG_SHAPE)
    model.load_weights(WEIGHT_FILE)

    # Prepare the input
    image = image.resize(IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    
    # Make prediction and compute the score
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    class_name = CLASS_NAMES[np.argmax(score)]
    confidence = 100*np.max(score)

    return class_name, confidence

if __name__ == '__main__':
    test_img = Image.open("app/data/biriyani.jpeg")
    class_name, score = classifier(test_img)
    print("Predicted : ", class_name, " with confidence : ", score)
