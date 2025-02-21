import os
import logging
import absl.logging
import shutil
import numpy as np
import cv2
from PIL import Image
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm # addng progress bar ui to console
import sys

# ignore warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
logging.getLogger("tensorflow").setLevel(logging.ERROR)
absl.logging.set_verbosity(absl.logging.ERROR)

# check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode = True, max_num_hands = 2, min_detection_confidence = 0.5)

# function to extarct hand regions from an image using MediaPipe hands
def extract_hand_landmarks(image):

    image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # convert PIL image to OpenCV format
    results = hands.process(image_rgb)  # detect hands

    hand_data = []

    if results.multi_hand_landmarks: # check if the hand can be detected with the landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z]) # normalized (x, y, z) coordinates - extracted from each landmark
            hand_data.append(landmarks)

    return hand_data # return list of hands, each with 21 (x, y, z) points

# function to load and process gif frames
# > 128 x 128 is used to prevent any memory errors with the GPU i am running on
# > 30 frames is a good balance for short gestures and gestures with continuous motion
def load_gif(file_path, max_frames = 30, target_size = (128, 128)):

    gif = Image.open(file_path)
    frames = []

    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor() # convert to CHW format (image data format in PyTorch) and normalizes it to [0, 1]
        # raw pixel values are 0 - 225 but deep leaning models work better when input values are normalized
        # normalized pixel = orginal pixel / 255
    ])

    hand_landmarks_data = [] # list to store hand landmarks for each frame

    try:
        while True:
            frame = gif.convert('RGB') # ensure 3-channel RGB since most machines learning models work with such images
            
            # extract hand landmarks from the current frame
            hand_landmarks = extract_hand_landmarks(frame)
            hand_landmarks_data.append(hand_landmarks)

            frame = transform(frame) # resize and normalize
            frames.append(frame) # store processed frames

            gif.seek(gif.tell() + 1) # move to the next frame

    except EOFError: # end of file is reached so stop
        pass

    # if its more than the max frames
    if len(frames) > max_frames:
        frames = frames[:max_frames]  # trim extra frames
        hand_landmarks_data = hand_landmarks_data[:max_frames]  # trim extra landmarks

    # pad sequences to ensure uniform shape (if less than max_frames)
    # need uniform shape for all sequences for them to b processed together in batches for machine learning
    while len(frames) < max_frames:
        frames.append(torch.zeros((3, target_size[0], target_size[1]))) # add blank frames
        hand_landmarks_data.append([]) # add empty landmarks for the blank frames

    return torch.stack(frames[:max_frames]), hand_landmarks_data # return as tensor (T, C, H, W) and hand landmarks data
    # tensor - image data represented in a multidimensional array

# custom dataset prepration - loading gifs and applying data augmentation
class SGSLDataset(Dataset):
    def __init__(self, folder_path, save_folder, max_frames = 30, target_size = (128, 128), augment = False):

        self.data = [] # store gif tensors
        self.labels = [] # store matching gesture labels
        self.hand_landmarks_data = [] # store hand landmarks
        self.class_mapping = {} # map gesure names to indicies for the data
        self.augment = augment # to allow data augmentaion or not
        self.save_folder = save_folder # folder to save augmented gifs
        self.augmentation_factor = 2 # no of augmented versions per sample

        label_counter = 0 # track label indices

        for gif_file in os.listdir(folder_path):
            if gif_file.endswith(".gif"):

                # extract gesture name from filename
                gesture_name = gif_file.split("_")[0] if "_" in gif_file else gif_file.split(".")[0]

                # assign a unique label index if not already assigned
                if gesture_name not in self.class_mapping:
                    self.class_mapping[gesture_name] = label_counter
                    label_counter += 1

                # load and process gif frames
                gif_path = os.path.join(folder_path, gif_file)
                frames, hand_landmarks = load_gif(gif_path, max_frames, target_size)

                # ensure if have hand landmarks
                if len(frames) == max_frames:
                    self.data.append(frames)
                    self.labels.append(self.class_mapping[gesture_name])
                    self.hand_landmarks_data.append(hand_landmarks)

        # convert to tensors - PyTorch models require tensor-based inputs
        self.data = torch.stack(self.data)
        self.labels = torch.tensor(self.labels, dtype = torch.long)

    # return the total no. of gifs in the dataset
    def __len__(self):

        if self.augment:
            return len(self.data) * self.augmentation_factor + len(self.data) # return with augmented data is applied + orginal data

        return len(self.data)
    
    def __getitem__(self, idx):
        
        # get gif tensor and corresponding class label
        gif = self.data[idx]
        label = self.labels[idx]
        hand_landmarks = self.hand_landmarks_data[idx] # get the corresponding hand landmarks

        return gif, label, hand_landmarks
    
    # create 3 additional versions of the frames with different augmentation settings
    def augment_data(self, frames):

        augmented_data = []

        # 3 different augmentation
        # augmentations = [
        #     transforms.Compose([
        #         transforms.RandomHorizontalFlip(p = 0.3),
        #         transforms.RandomRotation(degrees = 10),
        #         transforms.ColorJitter(brightness = 0.1, contrast = 0.1),
        #         transforms.Resize((112, 112))
        #     ]),
        #     transforms.Compose([
        #         transforms.RandomVerticalFlip(p = 0.3),
        #         transforms.RandomAffine(degrees = 5, translate = (0.01, 0.01)), # adjust orientation and position
        #         transforms.ColorJitter(saturation = 0.1, hue = 0.05),
        #         transforms.Resize((112, 112))
        #     ]),
        #     transforms.Compose([
        #         transforms.RandomRotation(degrees = 10),
        #         transforms.RandomHorizontalFlip(p = 0.5),
        #         transforms.RandomCrop(size = (100, 100)),  # random crop to simulate zoom
        #         transforms.Resize((112, 112))
        #     ])
        # ]

        augmentations = [
            transforms.Compose([
            transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2, hue = 0.1),
                    transforms.Resize((128, 128))
                ]),
                transforms.Compose([
                    transforms.RandomApply([transforms.ColorJitter(brightness = 0.3, contrast = 0.3)], p = 0.5),
                    transforms.RandomGrayscale(p = 0.2),
                    transforms.Resize((128, 128))
                ])
            ]


        # apply augmentation and add copies
        for aug in augmentations:
            augmented_frames = torch.stack([aug(frame) for frame in frames])  # apply augmentations per frame
            augmented_data.append(augmented_frames)

        return augmented_data
    
    def save_augmented_gif(self, frames, gesture_name, aug_idx):

        frames = [transforms.ToPILImage()(frame) for frame in frames] # convert each frame to a PIL index
        gif_name = f"{gesture_name}_augmented_{aug_idx}.gif"
        gif_path = os.path.join(self.save_folder, gif_name) # path will gif will be saved

        # save the first frame and append the rest to create a GIF
        frames[0].save(gif_path, save_all = True, append_images = frames[1:], loop = 0, duration = 100)

def augment_and_save_data(folder_path, save_folder):
    print("\nPreparing dataset...")
    dataset = SGSLDataset(folder_path, save_folder, augment = True)
    print(f"Total number of samples in the augmented dataset: {len(dataset)}\n")

    # iterate over the dataset and save the augmented GIFs
    for original_idx in tqdm(range(len(dataset.data)), desc = "Augmenting GIFs", unit = "gif"):
        original_frames = dataset.data[original_idx]  # original GIF frames
        gesture_name = list(dataset.class_mapping.keys())[list(dataset.class_mapping.values()).index(dataset.labels[original_idx].item())]

        # save the original gif as well into the new folder
        dataset.save_augmented_gif(original_frames, gesture_name, aug_idx = "original")

        augmented_versions = dataset.augment_data(original_frames)

        for aug_idx, aug_frames in enumerate(augmented_versions):
            dataset.save_augmented_gif(aug_frames, gesture_name, aug_idx)  # save augmented GIF

# CCN-Transformer hybrid model for sign language gesture recognition
# encapsulated in a class to group all related components (CNN, Transformer and classifiers) in one reusable structure
class CNNTransformerModel(nn.Module):
    def __init__(self, num_classes, feature_dim = 512, num_heads = 4, num_layers = 2, landmark_dim = 63):
        super(CNNTransformerModel, self).__init__()

        # CNN feature extractor (Resnet-18 backbone)
        self.cnn = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained = True) # make use of pretrained weights
        self.cnn.fc = nn.Identity() # remove final classification layer - get raw future embeddings to be processed by the transformer

        # transformer encoder - understand sequences of frames
        # feature dimension - larger values -> capture more complex features; match output dimension of cnn (512)
        # no. of attention heads - control the no. of attention mechanisms to process the sequence
        # no. of layers - defines the depth (more layers -> handle long term dependencies)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model = feature_dim, nhead = num_heads),
            num_layers = num_layers
        )

        # final classifier (fully connected layer) - maps feature vectors to class probabilities
        self.landmark_fc = nn.Linear(landmark_dim, feature_dim) # process hand landmarks
        self.fc = nn.Linear(feature_dim * 2, num_classes) # join both feature and landmarks

    # how the input tensor flows through the model
    def forward(self, x, hand_landmarks):
        batch_size, seq_len, c, h, w = x.shape # gesture sequences in a batch, no. of frame in sequence, channels, height and width of frame
        x = x.view(batch_size * seq_len, c, h, w) # reshaped for CNN to proces each frame individually

        # extract features using CNN
        features = self.cnn(x) # feature representation since final layer is removed
        features = features.view(batch_size, seq_len, -1) # reshaped for transformer - time-series of feature vectors

        # transformer encoding
        transformer_out = self.transformer(features) # learn temporal dependencies between frames

        # aggregate features - using mean pooling (for visual features)
        out = torch.mean(transformer_out, dim = 1) # combine multiple features into a single meaningful representation (dim = 1 -> average across all frames in the sequence)

        # process landmarks and combine with visual features
        landmark_features = self.landmark_fc(hand_landmarks) # process the hand landmarks
        combined_features = torch.cat((out, landmark_features), dim = -1)

        return self.fc(combined_features)

def collate_fn(batch):
    gifs, labels, landmarks = zip(*batch)
    gifs = torch.stack(gifs)  # ensure consistent shape
    labels = torch.tensor(labels, dtype = torch.long)
    return gifs, labels, landmarks

# function to load the dataset
def load_data(train_folder, test_folder, batch_size = 8, split_ratios = (0.85, 0.15)):
    print("\nLoading the dataset...")
    
    # load the dataset
    train_dataset = SGSLDataset(train_folder, save_folder = None, augment = False) # no need to augment anymore since data already augmented
    test_dataset = SGSLDataset(test_folder, save_folder = None, augment = False)

    # split the training and validation sets (85% - training, 15% - validation)
    train_size = int(split_ratios[0] * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_set, val_set = random_split(train_dataset, [train_size, val_size])

    # create dataloaders for model
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 4, pin_memory = True, collate_fn = collate_fn)
    val_loader = DataLoader(val_set, batch_size = batch_size, shuffle = False, num_workers = 4, pin_memory = True, collate_fn = collate_fn)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = 4, pin_memory = True, collate_fn = collate_fn)

    print("\nDone creating dataloaders for the model")

    return train_loader, val_loader, test_loader, len(train_dataset.class_mapping)

# function to train the model
def train_model(train_loader, val_loader, num_classes, epochs = 20, learning_rate = 0.001, save_path = "best_CNN_Transformer_model.pth"):

    print("\nTraining the model...")

    # setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # make use of gpu
    model = CNNTransformerModel(num_classes).to(device)

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss() # measure how well the model's predictions match the true labels
    optimizer = optim.Adam(model.parameters(), lr = learning_rate) # update the model's weight based on the computed loss

    # keep track of best validation accuracy
    best_val_acc = 0.0
    stats = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    # training epochs
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc = f"Epoch {epoch + 1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad() # reset gradient
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward() # compute gradient of the loss function
            optimizer.step() # apply computed gradients to adjust model's parameters
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader) # compute average training loss

        # conduct validation and get stats
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)
        stats['train_loss'].append(avg_train_loss)
        stats['val_loss'].append(val_loss)
        stats['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print("Best model saved!")

    with open('training_stats.txt', 'w') as f:
        for key, values in stats.items():
            f.write(f"{key}: {values}\n")

    print("\nFinished training the model")

    return model

# function to validate the model
def validate_model(model, val_loader, criterion, device):
    model.eval() # set the model to evaluation mode

    # mertics
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad(): # prevent from storing gradients - reduce memory usage
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device) # use gpu
            outputs = model(inputs) # model prediction
            # compute loss
            loss = criterion(outputs, labels) # cross entropy loss
            total_loss += loss.item()
            # compute predictions
            _, predicted = torch.max(outputs, 1) # get the most confident class
            # count correct prerdictions
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    # final predictions
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total

    return avg_loss, accuracy

# function to the test the model
def test_model(model, test_loader, device):
    model.load_state_dict(torch.load('best_CNN_Transformer_model.pth')) # load the saved model
    model.to(device) # use gpu
    model.eval() # set to evaluation
    correct = 0
    total = 0

    with torch.no_grad(): # not store gradients
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1) # extracts the predicted class with the highest probability
            correct += (predicted == labels).sum().item() # predictions to ground truth labels
            total += labels.size(0) # counts correctly classified

    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.4f}")
    
def app_menu():
    print("\nSelect an option:")
    print("1. Augment and Save Data")
    print("2. Train and Save Model")
    print("0. Exit")

def main():
    while True:

        # display application menu
        app_menu()

        choice = input("Enter your option: ")

        if choice == "1":
            folder_path = 'Data/ntu_sgsl'
            save_folder = 'Data/ntu_sgsl_augmented'

            if os.path.exists(save_folder):
                shutil.rmtree(save_folder)  # clear content of folder before adding data


            if not os.path.exists(save_folder):
                os.makedirs(save_folder)  # create folder if it doesn't exist

            augment_and_save_data(folder_path, save_folder)

            print("\nFinished processing and saving augmented images")

        elif choice == "2":
            # train_folder = 'Data/ntu_sgsl_augmented'
            train_folder = 'Data/ntu_sgsl_augmented'
            test_folder = 'Data/ntu_sgsl'
            train_loader, val_loader, test_loader, num_classes = load_data(train_folder, test_folder)

            model = train_model(train_loader, val_loader, num_classes)
            test_model(model, test_loader, torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            print("Testing complete!")

        elif choice == "0":
            sys.exit()

        else:
            print("\nInvalid input! Please enter a valid option.")

if __name__ == "__main__":
    main()


