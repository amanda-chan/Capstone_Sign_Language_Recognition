import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import Input, TimeDistributed, Dense, Flatten, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Functions to load and preprcess the NTU SGSL gif data
def load_gif(file_path):
    # Load the gif file, extract the indivdual frames, process the frames (resizing and normalizing pizel values), 
    # return the processed frames for input into a machine learning model
    gif = Image.open(file_path)
    frames = []
    try:
        while True:
            frame = gif.copy().convert('RGB')
            frame = frame.resize((112, 112))
            frames.append(np.array(frame, dtype = np.float32))
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    return np.array(frames) / 255.0  # Normalize pixel values

def load_gifs_from_folder(folder_path):
    data = []
    labels = []
    class_mapping = {}
    label_counter = 0

    for gif_file in os.listdir(folder_path):
        if gif_file.endswith(".gif"):
            # Extract the class - gesture name
            gesture_name = gif_file.split("_")[0] if "_" in gif_file else gif_file.split(".")[0]

            # Add gesture to class_mapping
            if gesture_name not in class_mapping:
                class_mapping[gesture_name] = label_counter
                label_counter += 1

            # Load gif frames
            gif_path = os.path.join(folder_path, gif_file)
            frames = load_gif(gif_path)
            data.append(frames)
            labels.append(class_mapping[gesture_name])

    return np.array(data, dtype = object), np.array(labels), class_mapping

# Function to build base model - CNN + Transformer Model
def build_model(input_shape, num_classes):
    # CNN Backbone
    cnn_base = ResNet50(weights = "imagenet", include_top = False, input_shape = (112, 112, 3))
    cnn_output = Flatten()(cnn_base.output) # Converts into 1D vector for easier processing
    cnn_model = Model(cnn_base.input, cnn_output)

    # TimeDistributed CNN
    # Allows the CNN to process each frame to extract spaital features
    frames_input = Input(shape=input_shape)
    time_distributed = TimeDistributed(cnn_model)(frames_input)

    # Transformer
    transformer = tf.keras.layers.MultiHeadAttention(num_heads = 4, key_dim = 64)(time_distributed, time_distributed)
    transformer = LayerNormalization()(transformer)
    transformer = Flatten()(transformer)

    # Classification Head
    output = Dense(num_classes, activation='softmax')(transformer)
    model = Model(frames_input, output)
    return model

#vLoad Data
folder_path = "Data/ntu_sgsl"
data, labels, class_mapping = load_gifs_from_folder(folder_path)

# Padding sequences to ensure uniform shape
max_frames = max(seq.shape[0] for seq in data)  # Maximum number of frames across all GIFs
data = np.array([np.pad(seq, ((0, max_frames - seq.shape[0]), (0, 0), (0, 0), (0, 0)), mode = 'constant') for seq in data])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2, random_state = 42)
y_train = to_categorical(y_train, num_classes = len(class_mapping))
y_test = to_categorical(y_test, num_classes = len(class_mapping))

# Build the Model
input_shape = (max_frames, 112, 112, 3)
num_classes = len(class_mapping)
model = build_model(input_shape=input_shape, num_classes=num_classes)

# Compile the Model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint("best_model.h5", save_best_only = True, monitor = 'val_accuracy', mode = 'max')
early_stopping = EarlyStopping(monitor = 'val_accuracy', patience = 5, restore_best_weights = True)

# Train the Model
history = model.fit(
    X_train, y_train,
    validation_split = 0.2,
    epochs = 50,
    batch_size = 8,
    callbacks = [checkpoint, early_stopping]
)

# Model Evaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size = 4)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")