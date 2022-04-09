import re
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import random
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50V2, VGG16
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image

from typing import Union

class TransferModel:
    def __init__(self, base: str, shape: tuple, classes: list, unfreeze: Union[list, str] = None):
        """
        Class for transfer learning from either VGG16 or ResNet
        Args:
            base: String giving the name of the base model (either 'VGG16' or 'ResNet')
            shape: Input shape as tuple (height, width, channels)
            classes: List of class labels
        """
        self.shape = shape
        self.classes = classes
        self.history = None
        self.base = base
        self.model = None
        self.freeze = None

        # Class allows for two base models (VGG16 oder ResNet)
        # Use pre-trained ResNet model
        if self.base == 'ResNet':
            self.base_model = ResNet50V2(include_top=False, input_shape=self.shape, weights='imagenet')#weights=None)

            self.base_model.trainable = False
            if unfreeze is not None:
                self.base_model = self._make_trainable(model=self.base_model, patterns=unfreeze)

            add_to_base = self.base_model.output
            add_to_base = GlobalAveragePooling2D(data_format='channels_last', name='head_gap')(add_to_base)

        # Use pre-trained VGG16
        elif self.base == 'VGG16':
            self.base_model = VGG16(include_top=False, input_shape=self.shape, weights='imagenet')

            self.base_model.trainable = False
            if unfreeze is not None:
                self.base_model = self._make_trainable(model=self.base_model, patterns=unfreeze)

            add_to_base = self.base_model.output
            add_to_base = Flatten(name='head_flatten')(add_to_base)
            add_to_base = Dense(1024, activation='relu', name='head_fc_1')(add_to_base)
            add_to_base = Dropout(0.3, name='head_drop_1')(add_to_base)
            add_to_base = Dense(1024, activation='relu', name='head_fc_2')(add_to_base)
            add_to_base = Dropout(0.3, name='head_drop_2')(add_to_base)

        # Add final output layer
        new_output = Dense(len(self.classes), activation='softmax', name='head_pred')(add_to_base)
        self.model = Model(self.base_model.input, new_output)

        # Model overview
        layers = [(layer, layer.name, layer.trainable) for layer in self.model.layers]
        self.freeze = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])

    @staticmethod
    def _make_trainable(model, patterns: Union[str, list]):
        """
        Helper function to make certain (or even all) layers trainable
        Args:
            model: tf.Model
            patterns: list or str == 'all', of patterns as str to match layer names
        Returns:
            model
        """
        if isinstance(patterns, str) and patterns == 'all':
            model.trainable = True
        else:
            for layer in model.layers:
                for pattern in patterns:
                    regex = re.compile(pattern)
                    if regex.search(layer.name):
                        layer.trainable = True
                    else:
                        pass

        return model

    def load(self, path: str):
        """
        Load a trained model into self.model
        Args:
            path: Path to directory containing saved model
        Returns:
            Nothing
        """
        if not path.endswith("/"):
            path += "/"

        file = open(path + "classes.pickle", 'rb')
        self.classes = pickle.load(file)
        self.model = tf.keras.models.load_model(path + "model")

    def save(self, folderpath: str):
        """
        Save the model using tf.keras.model.save
        Args:
            folderpath: (Full) Filepath to store model
        """

        # Make sure folderpath ends on slash, else fix
        if not folderpath.endswith("/"):
            folderpath += "/"

        if self.model is not None:
            os.mkdir(folderpath)
            model_path = folderpath + "model"
            # Save model to model dir
            self.model.save(filepath=model_path)
            # Save associated class mapping
            class_df = pd.DataFrame({'classes': self.classes})
            class_df.to_pickle(folderpath + "classes.pickle")
        else:
            raise AttributeError('Model does not exist')

    def compile(self, **kwargs):
        """
        Compile method
        """
        self.model.compile(**kwargs)

    def train(self, ds_train: tf.data.Dataset, epochs: int, ds_valid: tf.data.Dataset = None, class_weights: np.array = None, model_label = random.randint(0,10000)):
        """
        Trains model in ds_train with for epochs rounds
        Args:
            ds_train: training data as tf.data.Dataset
            epochs: number of epochs to train
            ds_valid: optional validation data as tf.data.Dataset
            class_weights: optional class weights to treat unbalanced classes
        Returns
            Training history from self.history
        """

        # Define early stopping as callback
        # early_stopping = EarlyStopping(monitor='categorical_accuracy', min_delta=0.05, patience=1, restore_best_weights=False)
        
        checkpoint_filepath = 'checkpoints/' + model_label
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                    filepath=checkpoint_filepath,
                                    save_weights_only=False,
                                    monitor='categorical_accuracy',
                                    mode='max',
                                    save_best_only=True)


        callbacks = [model_checkpoint_callback]

        # Fitting
        self.history = self.model.fit(ds_train, epochs=epochs, validation_data=ds_valid, callbacks=callbacks, class_weight=class_weights)

        return self.history

    def evaluate(self, ds_test: tf.data.Dataset):
        """
        Evaluate model on ds_test
        Args:
            ds_test: Testing data as tf.data.Dataset
        Returns:
              Dictionary with metrics and values on ds_test
        """

        result = self.model.evaluate(ds_test)

        if isinstance(result, int):
            return {'loss': result}
        else:
            return dict(zip(self.model.metrics_names, result))

    def predict(self, ds_new: tf.data.Dataset, proba: bool = True):
        """
        Predict class probs or labels on ds_new
        Labels are obtained by taking the most likely class given the predicted probs
        Args:
            ds_new: New data as tf.data.Dataset
            proba: Boolean if probabilities should be returned
        Returns:
            class labels or probabilities
        """

        p = self.model.predict(ds_new)

        if proba:
            return p
        else:
            return [np.argmax(x) for x in p]

    def predict_from_jpeg_path(self, filepath, classes):
        """
        Method to predict given a filepath to .jpeg as input
        Args:
          filepath: Path to file we want prediction for
          classes:
        Returns:
          prediction class label
        """
        img = image.load_img(filepath, target_size=(224, 224))
        img = image.img_to_array(img)
        img /= 255.0
        img = img.reshape(-1, *img.shape)
        pred = self.predict(img, proba=False)
        return classes[pred[0]]

    def predict_from_array(self, img, classes):
        """
        Method to predict given an array input
        Args:
          img: Image array with shape (224, 224, 3) with values in [0, 1]
          classes:
        Returns:
          prediction class label
        """

        # Reshape image to include additional dimension needed for prediction
        img = img.reshape(-1, *img.shape)
        pred = self.predict(img, proba=False)
        return classes[pred[0]]

    def plot(self, what: str = 'metric'):
        """
        Show a visualization of training and validation process
        Args:
            what: Plot training loss or metric?
        """
        if self.history is None:
            AttributeError("No training history available, call TransferModel.train first")
    
        if what not in ['metric', 'loss']:
            AttributeError(f'what must be either "loss" or "metric"')
    
        if what == 'metric':
            metric = self.model.metrics_names[1]
            y_1 = self.history.history[metric]
            y_2 = self.history.history['val_' + metric]
            y_label = metric
        # elif what == 'loss':
            y_3 = self.history.history['loss']
            y_4 = self.history.history['val_loss']
            y_label2 = 'loss'

        plt.figure("Accuracy")
        plt.plot(y_1)
        plt.plot(y_2)
        plt.title('Model Performance')
        plt.ylabel(y_label)
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.figure("Loss")
        plt.plot(y_3)
        plt.plot(y_4)
        plt.title('Model Loss')
        plt.ylabel(y_label2)
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()