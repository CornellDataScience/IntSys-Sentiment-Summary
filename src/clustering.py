from keras.backend.tensorflow_backend import set_session
from keras.layers import Dense, Dropout
from keras.models import Model
from keras.models import Sequential
from keras import backend as K

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python.framework.errors_impl import ResourceExhaustedError

import numpy as np
from sklearn.cross_decomposition import CCA

# CONSTANTS
NUM_EPOCHS = 200
BATCH_SIZE = 256

K_COMPS = 3
MAX_ITER = 1500

# TENSORFLOW GPU LIMITS
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                   # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

def create_helpfulness_classifier(input_dim, num_classes, output_dim_lst, dropout_lst):
    star_classifier = Sequential()
    star_classifier.add(Dense(output_dim_lst[0], input_dim=input_dim, activation='relu'))
    #star_classifier.add(Dropout(dropout_lst[0]))
    star_classifier.add(Dense(output_dim_lst[1], activation='relu'))
    #star_classifier.add(Dropout(dropout_lst[1]))
    star_classifier.add(Dense(output_dim_lst[2], activation='relu'))
    star_classifier.add(Dense(num_classes, activation='softmax'))

    star_classifier.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return star_classifier

# compiled model
# train_y is list of one-hot vectors
def train_model(model, train_x, train_y):
    model.fit(train_x, train_y, 
              epochs = NUM_EPOCHS, batch_size = BATCH_SIZE, verbose = 1)

def perform_CCA(sent_embeddings, signal_embeddings):
    cca = CCA(n_components=K_COMPS, max_iter=MAX_ITER)
    cca.fit(sent_embeddings[:1000], signal_embeddings[:1000])

    sent_k, sig_k = cca.transform(sent_embeddings, signal_embeddings)
    return sent_k, sig_k

# (centroids, lst of idxs closest to centroid)
def compute_centroids(low_dim_embeddings, val_y):
    classes = list(set(val_y))
    classes.sort()

    class_centroids = np.zeros((len(classes), low_dim_embeddings.shape[1]))
    close_to_centroids = []

    for i, c in enumerate(classes):
        class_low_dim_embeddings = low_dim_embeddings[val_y == i, :] 
        class_centroids[i, :] = np.mean(class_low_dim_embeddings)

        class_normed = np.sum(class_low_dim_embeddings**2,axis=1)**(1./2)
        close_to_centroids.append(np.argmin(class_normed))

    return class_centroids, close_to_centroids

