import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import efficientnet.tfkeras as efn
import pickle
import numpy as np
with open(r'models\lgbm.pickle', 'rb') as handle:
    lgb = pickle.load(handle)
def metadata(sex,age,site):
    site = np.nan if site==6 else site
    pred = lgb.predict(np.array([sex, age, site]).reshape(
        1, 3), num_iteration=lgb.best_iteration)[0]
    return pred
def load_model(path):
    model_input = tf.keras.Input(shape=(224, 224, 3), name='imgIn')

    dummy = tf.keras.layers.Lambda(lambda x: x)(model_input)

    outputs = []
    constructor = getattr(efn, f'EfficientNetB0')

    x = constructor(include_top=False, weights=None,
                    input_shape=(224, 224, 3),
                    pooling='avg')(dummy)

    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    outputs.append(x)

    model = tf.keras.Model(model_input, outputs, name='aNetwork')
    model.load_weights(path)
    return model

def prepare_image(path):
    image_raw = tf.io.read_file(path)
    img = tf.image.decode_image(image_raw)
    img = tf.image.resize(img, [256, 256])
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.image.central_crop(img, 250 / 256)
    img = tf.image.resize(img, [224, 224])
    img = tf.reshape(img, [1,224, 224, 3])
    return img

def predict(path,melanoma,age,site,gender):
    x = prepare_image(path)
    pred = 0.9*melanoma.predict(x)[0][0] + 0.1*metadata(gender, age, site)
    result = f"{(pred)*100:.2f} % sure that this is Malignant" if pred > 0.5 else f"{(1 - pred)*100:.2f} % sure that this is Benign"
    return result
if __name__ == "__main__":
    melanoma = load_model(r'models\model.h5')