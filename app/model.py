import tensorflow as tf

from tensorflow.keras import layers

IMG_SIZE = (224, 224)
CHANNELS = 3
IMG_SHAPE = IMG_SIZE+(CHANNELS,)
BATCH_SIZE = 32
CLASS_NAMES = ('Apple Pie', 'Biryani', 'Chicken Curry', 'Cupcake', 'Eggs Denedict')
WEIGHT_FILE = "./weights/best_model_vgg19_2_90val.hdf5"


def vgg19_modified(in_shape, vgg_base_weights=None, class_count = 5):
    """
    Returns a model built arount VGG19 architecture.
    in_shape : input shape, a tuple (height, width, channels)
    vgg_base_weights : how to initialize vgg19 base model. Possible values : None, 'imagenet' or path to weightfile
    class_count : number of classes (default = 5)
    Returns a modified version og VGG19
    """
    base_model = tf.keras.applications.VGG19(
        input_shape=in_shape,
        include_top=False,
        weights=vgg_base_weights)
    
    # Base model specific pre-processing
    base_model.trainable = False

    preprocess = tf.keras.applications.vgg19.preprocess_input

    # Data augmentation
    data_augmentation = tf.keras.Sequential([layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.1)],)
    
    # New layers
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    dense_layer = layers.Dense(512, activation='relu')
    drop_out_layer = layers.Dropout(0.2)
    prediction_layer = layers.Dense(class_count)

    # assemble the model
    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = data_augmentation(inputs)
    x = preprocess(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = dense_layer(x)
    x = drop_out_layer(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    return model
