import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Print the shape of the data
print("Original Data Shape:", data.shape)  # This will help to understand the data

# Reshaping the data into a 7x6 grid (since 7 * 6 = 42)
N_samples = data.shape[0]
height, width = 7, 6  # Target image size (7x6)

# Reshape data into (N_samples, 7, 6, 1) for grayscale images
data_reshaped = data.reshape(N_samples, height, width, 1)

# One-hot encoding labels
labels_one_hot = to_categorical(labels)

# Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data_reshaped, labels_one_hot, test_size=0.2, shuffle=True, stratify=labels)

# Define CNN model
def build_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))  # Output layer
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Build and train CNN model
cnn_model = build_cnn_model(x_train.shape[1:])
cnn_model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1)

# Evaluate the model
score = cnn_model.evaluate(x_test, y_test, verbose=1)
print(f"Test loss: {score[0]}")
print(f"Test accuracy: {score[1]}")

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': cnn_model}, f)



