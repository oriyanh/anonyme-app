import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D

from easyfacenet.simple import facenet


# class SubstituteModel(Model):
#
#     def __init__(self, num_classes):
#         self.conv1 = Conv2D(64, 2, 2)
#         self.conv2 = Conv2D(64, 2, 2)
#         self.dense1 = Dense(200, activation='relu')
#         self.dense2 = Dense(200, activation='relu')
#         self.dense3 = Dense(num_classes, activation='softmax')
#
#     def call(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.dense1(x)
#         x = self.dense2(x)
#         return self.dense3(x)

class SubstituteModel(Model):

    def __init__(self, num_classes):
        self.dense = Dense(num_classes, activation='softmax')

    def call(self, x):
        return self.dense(x)

def get_embeddings(images):
    return facenet.embedding(images)

def classify(model, images):
    embeddings = get_embeddings(images)
    return model(embeddings)