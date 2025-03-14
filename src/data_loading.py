import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import ssl

def load_data():
    # Désactiver temporairement la vérification SSL
    ssl._create_default_https_context = ssl._create_unverified_context

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    return (x_train, y_train), (x_test, y_test)

