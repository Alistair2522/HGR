# train_model.py
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Lambda
from tensorflow.keras.utils import to_categorical, register_keras_serializable
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -------------------------------------------------
# Define and register the custom background subtraction layer
# -------------------------------------------------
@register_keras_serializable()
class BackgroundSubtractionLayer(tf.keras.layers.Layer):
    def __init__(self, background, **kwargs):
        """
        background: a NumPy array of shape (height, width, channels)
        """
        super(BackgroundSubtractionLayer, self).__init__(**kwargs)
        self.background = tf.constant(background, dtype=tf.float32)

    def call(self, inputs):
        return inputs - self.background

    def get_config(self):
        config = super().get_config().copy()
        # Save the background as a list (for serialization)
        config.update({'background': self.background.numpy().tolist()})
        return config

    @classmethod
    def from_config(cls, config):
        background = np.array(config.pop('background'))
        return cls(background=background, **config)

# -------------------------------------------------
# Load and preprocess data
# -------------------------------------------------
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])
print("Original Data Shape:", data.shape)  # e.g., (N_samples, 42)

# Reshape the flattened vectors (of length 42) into 7x6 grayscale images.
N_samples = data.shape[0]
height, width = 7, 6  # since 7*6 = 42
data_reshaped = data.reshape(N_samples, height, width, 1).astype('float32')

# Normalize pixel values to [0, 1]
data_reshaped /= 255.0

# One-hot encode the labels
labels_one_hot = to_categorical(labels)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    data_reshaped, labels_one_hot, test_size=0.2, shuffle=True, stratify=labels
)

# -------------------------------------------------
# Data Augmentation
# -------------------------------------------------
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(x_train)

# -------------------------------------------------
# Compute a Background Image
# -------------------------------------------------
# Compute the background as the mean of the training images.
background = np.mean(x_train, axis=0)  # shape: (7, 6, 1)

# -------------------------------------------------
# Build the CNN Model (with the custom background subtraction layer)
# -------------------------------------------------
def build_cnn_model(input_shape, background):
    model = Sequential()
    # Optionally add a normalization layer if needed
    # model.add(Lambda(lambda x: x / tf.reduce_max(x), input_shape=input_shape, name='input_normalization'))
    
    # The first layer subtracts the background from each input image.
    model.add(BackgroundSubtractionLayer(background, input_shape=input_shape, name='background_subtraction_layer'))
    
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

cnn_model = build_cnn_model(x_train.shape[1:], background)

# -------------------------------------------------
# Callbacks for Early Stopping and Best Model Checkpointing
# -------------------------------------------------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

]

# -------------------------------------------------
# Train the Model
# -------------------------------------------------
batch_size = 32
epochs = 20
history = cnn_model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    epochs=epochs,
    validation_data=(x_test, y_test),
    callbacks=callbacks,
    verbose=1
)

score = cnn_model.evaluate(x_test, y_test, verbose=1)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# -------------------------------------------------
# Save the Trained Model using Keras’ native saving method
# -------------------------------------------------
cnn_model.save('my_model.keras')

