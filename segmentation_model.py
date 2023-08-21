import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import cv2
from PIL import Image
import numpy as np
from patchify import patchify
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from matplotlib import pyplot as plt
import random

# Import necessary libraries for data processing
from tensorflow.keras.utils import to_categorical
# Import train_test_split from sklearn
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.layers import concatenate, BatchNormalization, Dropout, Lambda
from keras import backend as K

import segmentation_models as sm

import tensorflow as tf
import tensorflow_hub as hub

import time





def check_modules():
    #Make sure all neccessary models exist
    modules = ['os', 'cv2', 'numpy', 'patchify','sklearn', 'matplotlib', 'random', 'tensorflow', 'segmentation-models', 'keras']

    missing = []
    import importlib.util
    import sys

    for i in modules:
        spec = importlib.util.find_spec(i)
        if spec is None:
            print(i +" is not installed")
            missing.append(i)
    if len(missing) == 0:
        print("All necessary packages installed!")
    else:
        print("Please install missing packages mentioned above before proceeding")


def system_test():
    drive_test = input("Are you running this code from a google colab? [y/n]:" )

    if drive_test == 'y':
        print("\nPlease provide path to the 'data' folder in this directory.")
        print("(hint: for most uers it will likely be: '/content/drive/MyDrive/Semantic segmentation/data/')")
        dataset_root_folder = input("Data directory path:" )

    if drive_test == 'n':
        root = os.getcwd()
        dataset_root_folder = root + '\data'
        print(f"\nRoot folder identified as: {dataset_root_folder}")
        test_root = input("\nDoes the above path correctly reflect the location of the training data? [y/n]: ")
        if test_root == 'n':
            print("\nPlease manually input the path of the training data folder on your local machine")
            dataset_root_folder = input("Data direcotry path:")


    print(f"\nDataset folder path set to {dataset_root_folder}")
    return dataset_root_folder



def test_extensions(dataset_root_folder):
    print("Testing all file extensions are correct....")

    incorrect_extensions = []
    ### Loop through the directory tree using os.walk()
    ### os.walk() =>Do something with the current directory information (dataset_root_folder)
    for path, subdirs, files in os.walk(dataset_root_folder):
        ### Extract the name of the current directory
        dir_name = path.split(os.path.sep)[-1]
        ### Check if the directory is named 'masks'
        if dir_name == 'masks': # 'images
        ### List the files in the current directory
            images = os.listdir(path)
        ### Loop through the images in the current directory if its neded another extention just change
            for i, image_name in enumerate(images):
                ### Check if the image has a '.png' extension
                if (image_name.endswith('.png')): # '.jpg
                    pass
                else:
                    print(f"Incorrect file extension for: {image_name}")
                    incorrect_extensions.append(image_name)

    if len(incorrect_extensions) > 0:
        print("Please fix the above image extensions before proceeding")
        

    
    
def preprocess(dataset_root_folder):
    minmaxscaler = MinMaxScaler()
    
    ### Define the patch size for images
    image_patch_size = 256
    print(f"Setting image patch size to {image_patch_size}")



    ### Initialize lists to store image and mask datasets
    image_dataset = []
    mask_dataset = []


    print("Creating patches for masks and images...")
    # Loop through different image types ('images' and 'masks')
    for image_type in ['images' , 'masks']:
      # Determine the image extension based on the image type
        if image_type == 'images':
            image_extension = 'jpg'
        elif image_type == 'masks':
             image_extension = 'png'
      ### Loop through tile IDs and image IDs
        for tile_id in range(1,8):
            for image_id in range(1,20):
                ### Read an image using OpenCV
                image = cv2.imread(f'{dataset_root_folder}/Tile {tile_id}/{image_type}/image_part_00{image_id}.{image_extension}',1)
                ### Check if the image is not None
                if image is not None:
                ### Convert mask images from BGR to RGB
                    if image_type == 'masks':
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    ### Calculate size_x and size_y for cropping
                    size_x = (image.shape[1]//image_patch_size)*image_patch_size
                    size_y = (image.shape[0]//image_patch_size)*image_patch_size
                    ### Convert the image to a PIL Image object
                    image = Image.fromarray(image)
                    ### Crop the image
                    image = image.crop((0,0, size_x, size_y))
                    ### Convert the PIL Image back to a numpy array
                    image = np.array(image)
                    ### Create image patches using 'patchify'
                    patched_images = patchify(image, (image_patch_size, image_patch_size, 3), step=image_patch_size)
                    #print(len(patched_images))
                    ### Loop through the patches and add them to the datasets
                    for i in range(patched_images.shape[0]):
                        for j in range(patched_images.shape[1]):
                            if image_type == 'images':
                                ### Get an individual image patch
                                individual_patched_image = patched_images[i,j,:,:]
                                ### Apply Min-Max scaling to the image patch
                                individual_patched_image = minmaxscaler.fit_transform(individual_patched_image.reshape(-1, individual_patched_image.shape[-1])).reshape(individual_patched_image.shape)
                                ### Get the scaled image patch
                                individual_patched_image = individual_patched_image[0]
                                ### Add the image patch to the image dataset
                                image_dataset.append(individual_patched_image)
                            elif image_type == 'masks':
                                ### Get an individual mask patch
                                individual_patched_mask = patched_images[i,j,:,:]
                                individual_patched_mask = individual_patched_mask[0]
                                ### Add the mask patch to the mask dataset
                                mask_dataset.append(individual_patched_mask)




    print("Converting datasets to numpy arrays...")
    # Convert the datasets to numpy arrays
    image_dataset = np.array(image_dataset)
    mask_dataset = np.array(mask_dataset)

    ### Print the lengths of image and mask datasets
    print(f"The final length of of the image dataset is : {len(image_dataset)}")
    print(f"The final length of the mask dataset is {len(mask_dataset)}")

    if len(image_dataset) != len(mask_dataset):
        sys.exit("WARNING: Image and mask dataset do not have the same length, something is wrong")
        
    return image_dataset, mask_dataset


def define_colors():
    print("Defining color values for different classes...")
    # Define color values for different classes
    class_building_hex = '#3C1098'
    class_building = class_building_hex.lstrip('#')
    class_building = np.array(tuple(int(class_building[i:i+2], 16) for i in (0,2,4)))
    # print(class_building)
    print(f"\tThe BUILDING class corresponds to the hex code: {class_building_hex}")

    class_land_hex = '#8429F6'
    class_land = class_land_hex.lstrip('#')
    class_land = np.array(tuple(int(class_land[i:i+2], 16) for i in (0,2,4)))
    print(f"\tThe LAND class corresponds to the hex code: {class_land_hex}")

    class_road_hex = '#6EC1E4'
    class_road = class_road_hex.lstrip('#')
    class_road = np.array(tuple(int(class_road[i:i+2], 16) for i in (0,2,4)))
    print(f"\tThe ROAD class corresponds to the hex code: {class_road_hex}")

    class_vegetation_hex = '#FEDD3A'
    class_vegetation = class_vegetation_hex.lstrip('#')
    class_vegetation = np.array(tuple(int(class_vegetation[i:i+2], 16) for i in (0,2,4)))
    print(f"\tThe VEGETATION class corresponds to the hex code: {class_vegetation_hex}")

    class_water_hex = '#E2A929'
    class_water = class_water_hex.lstrip('#')
    class_water = np.array(tuple(int(class_water[i:i+2], 16) for i in (0,2,4)))
    print(f"\tThe WATER class corresponds to the hex code: {class_water_hex}")

    class_unlabeled_hex = '#9B9B9B'
    class_unlabeled = class_unlabeled_hex.lstrip('#')
    class_unlabeled = np.array(tuple(int(class_unlabeled[i:i+2], 16) for i in (0,2,4)))
    print(f"\tThe UNLABELED class corresponds to the hex code: {class_unlabeled_hex}")
    
    return class_building, class_land, class_road, class_vegetation, class_water, class_unlabeled
    
    

# Convert RGB labels to class labels using predefined color values


def create_labels(mask_dataset):
    class_building, class_land, class_road, class_vegetation, class_water, class_unlabeled = define_colors()
    def rgb_to_label(label):
        label_segment = np.zeros(label.shape, dtype=np.uint8)
        label_segment[np.all(label == class_water, axis=-1)] = 0
        label_segment[np.all(label == class_land, axis=-1)] = 1
        label_segment[np.all(label == class_road, axis=-1)] = 2
        label_segment[np.all(label == class_building, axis=-1)] = 3
        label_segment[np.all(label == class_vegetation, axis=-1)] = 4
        label_segment[np.all(label == class_unlabeled, axis=-1)] = 5
        #print(label_segment)
        label_segment = label_segment[:,:,0]
        #print(label_segment)
        return label_segment
    print("Converting the mask dataset to correspond with class labels")
    # Convert mask dataset to class labels
    labels = []
    for i in range(mask_dataset.shape[0]):
        label = rgb_to_label(mask_dataset[i])
        labels.append(label)

    # Convert labels to a numpy array
    labels = np.array(labels)

    # Expand dimensions of the labels array
    labels = np.expand_dims(labels, axis=3)
    
    return labels
      

    
def visualize_mask_image(image_dataset, labels):
    print("\nBelow is a random image and its mask")
    # Generate a random image ID
    random_image_id = random.randint(0, len(image_dataset))

    # Plot a random image and its corresponding label
    plt.figure(figsize=(14,8))
    plt.subplot(121)
    plt.imshow(image_dataset[random_image_id])
    plt.subplot(122)
    #plt.imshow(mask_dataset[random_image_id])
    plt.imshow(labels[random_image_id][:,:,0])
    plt.show()


    preview_again = input("Would you like to preview another random image? [y/n]: ")
    while preview_again == "y":
        # Generate a random image ID
        random_image_id = random.randint(0, len(image_dataset))


        # Plot a random image and its corresponding label
        plt.figure(figsize=(14,8))
        plt.subplot(121)
        plt.imshow(image_dataset[random_image_id])
        plt.subplot(122)
        #plt.imshow(mask_dataset[random_image_id])
        plt.imshow(labels[random_image_id][:,:,0])
        plt.show()

        preview_again = input("Would you like to preview another random image? [y/n]: ")
        

def run_preprocessing():
    dataset_root_folder = system_test()
    test_extensions(dataset_root_folder)
    
    image_dataset, mask_dataset = preprocess(dataset_root_folder)
    
    labels = create_labels(mask_dataset)
    
    visualize_mask_image(image_dataset, labels)
    
    print("Done creating and pre-processing dataset!")
    
    return image_dataset, labels

#### BEGINNING MODEL SET UP#####

def create_training_data(labels, image_dataset):
    total_classes = len(np.unique(labels))
    # Convert labels to categorical format
    labels_categorical_dataset = to_categorical(labels, num_classes=total_classes)
    # Assign the image dataset to the master training dataset
    master_trianing_dataset=image_dataset
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(master_trianing_dataset, labels_categorical_dataset, test_size=0.15, random_state=100)
    
    return X_train, X_test, y_train, y_test


#####
# Define a function for the Jaccard coefficient
def jaccard_coef(y_true, y_pred):
  y_true_flatten = K.flatten(y_true)
  y_pred_flatten = K.flatten(y_pred)
  intersection = K.sum(y_true_flatten * y_pred_flatten)
  final_coef_value = (intersection + 1.0) / (K.sum(y_true_flatten) + K.sum(y_pred_flatten) - intersection + 1.0)
  return final_coef_value


# Define a function to create the UNet-like model
def multi_unet_model(n_classes=5, image_height=256, image_width=256, image_channels=1):

  # Define the input layer
  inputs = Input((image_height, image_width, image_channels))

  source_input = inputs

  c1 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(source_input)
  c1 = Dropout(0.2)(c1)
  c1 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c1)
  p1 = MaxPooling2D((2,2))(c1)

  c2 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p1)
  c2 = Dropout(0.2)(c2)
  c2 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c2)
  p2 = MaxPooling2D((2,2))(c2)

  c3 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p2)
  c3 = Dropout(0.2)(c3)
  c3 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c3)
  p3 = MaxPooling2D((2,2))(c3)
  c4 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p3)
  c4 = Dropout(0.2)(c4)
  c4 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c4)
  p4 = MaxPooling2D((2,2))(c4)

  c5 = Conv2D(256, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p4)
  c5 = Dropout(0.2)(c5)
  c5 = Conv2D(256, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c5)

  u6 = Conv2DTranspose(128, (2,2), strides=(2,2), padding="same")(c5)
  u6 = concatenate([u6, c4])
  c6 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u6)
  c6 = Dropout(0.2)(c6)
  c6 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c6)

  u7 = Conv2DTranspose(64, (2,2), strides=(2,2), padding="same")(c6)
  u7 = concatenate([u7, c3])
  c7 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u7)
  c7 = Dropout(0.2)(c7)
  c7 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c7)
  u8 = Conv2DTranspose(32, (2,2), strides=(2,2), padding="same")(c7)
  u8 = concatenate([u8, c2])
  c8 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u8)
  c8 = Dropout(0.2)(c8)
  c8 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c8)

  u9 = Conv2DTranspose(16, (2,2), strides=(2,2), padding="same")(c8)
  u9 = concatenate([u9, c1], axis=3)
  c9 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u9)
  c9 = Dropout(0.2)(c9)
  c9 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c9)

  # Define the output layer
  outputs = Conv2D(n_classes, (1,1), activation="softmax")(c9)

  # Create the model
  model = Model(inputs=[inputs], outputs=[outputs])
  return model




def create_loss_function():
    
    weights = [0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]
    dice_loss = sm.losses.DiceLoss(class_weights = weights)
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)
    return total_loss


def create_deep_learning_model(labels, image_dataset):
    
    print("Partitioning training and testing data...")
    
    X_train, X_test, y_train, y_test = create_training_data(labels, image_dataset)
    
    # Get the height, width, channels, and number of classes from the dataset
    image_height = X_train.shape[1]
    image_width = X_train.shape[2]
    image_channels = X_train.shape[3]
    total_classes = y_train.shape[3]

    # Print the shapes of the training and testing sets
    print(f"\tThere are {X_train.shape[0]} images in the training dataset")
    print(f"\tThere are {X_test.shape[0]} images in the test dataset")
    print(f"\tThe images are {image_height} by {image_width} pixels with {image_channels} channels. The masks contain {total_classes} categorical classes")
    
    
    print("Instantiating Unet model....")
    model = multi_unet_model(n_classes=total_classes,
                          image_height=image_height,
                          image_width=image_width,
                          image_channels=image_channels)
         
    
    total_loss = create_loss_function() 
    metrics = ["accuracy", jaccard_coef]
    print("Compiling model...")
    model.compile(optimizer="adam", loss=total_loss, metrics=metrics)
    
    preview_model = input("\nWould you like to view a summary of the model? [y/n]:")
    
    if preview_model == 'y':
        model.summary()
        
    return model, X_train, X_test, y_train, y_test

def customizable_training_params():
    
    potential_batch_sizes = [1, 2, 4, 8, 16, 32, 64]

    batch_size = 1
    epochs = 5
    change_batch = input(f"\nThe default batch size is {batch_size}. Would you like to change it? [y/n]:")
    if change_batch == "y":
        print(f"Please choose a desired batch size from the following values: {potential_batch_sizes}")
        input_batch_size = int(input("\nChoose desired batch size:"))
        while input_batch_size not in potential_batch_sizes:
            print(f"Input batch size of {input_batch_size} not an accepted value.")
            print(f"Please choose a desired batch size from the following values: {potential_batch_sizes}")
            input_batch_size = int(input("Choose desired batch size:"))
        batch_size = input_batch_size

    change_epoch = input(f"\nThe default number of training epochs is {epochs}. Would you like to change it? [y/n]:")
    if change_epoch == "y":
        print("\nPlease choose a new number of training epochs.")
        print("IMPORTANT: Too many epochs will substantially elongate training time and may lead to overfitting. \nIt is recommended to choose a value between 5 and 20 epochs.")
        epochs = int(input("\nNew number of training epochs (integers only): "))
    
    return batch_size, epochs
    
    
       

def test_model(model, X_test, y_test):
    
    print("\n\n Preparing visual preview of model prediction on random test image...")
    
    y_test_argmax = np.argmax(y_test, axis=3)
    test_image_number = random.randint(0, len(X_test))

    test_image = X_test[test_image_number]
    ground_truth_image = y_test_argmax[test_image_number]

    test_image_input = np.expand_dims(test_image, 0)

    prediction = model.predict(test_image_input)
    predicted_image = np.argmax(prediction, axis=3)
    predicted_image = predicted_image[0,:,:]
    
    plt.figure(figsize=(14,8))
    plt.subplot(231)
    plt.title("Original Image")
    plt.imshow(test_image)
    plt.subplot(232)
    plt.title("Original Masked image")
    plt.imshow(ground_truth_image)
    plt.subplot(233)
    plt.title("Predicted Image")
    plt.imshow(predicted_image)
    plt.show()
    
    preview_again = input("Would you like to preview prediction results on another random image? [y/n]: ")
    while preview_again == "y":
        test_image_number = random.randint(0, len(X_test))

        test_image = X_test[test_image_number]
        ground_truth_image = y_test_argmax[test_image_number]

        test_image_input = np.expand_dims(test_image, 0)

        prediction = model.predict(test_image_input)
        predicted_image = np.argmax(prediction, axis=3)
        predicted_image = predicted_image[0,:,:]

        plt.figure(figsize=(14,8))
        plt.subplot(231)
        plt.title("Original Image")
        plt.imshow(test_image)
        plt.subplot(232)
        plt.title("Original Masked image")
        plt.imshow(ground_truth_image)
        plt.subplot(233)
        plt.title("Predicted Image")
        plt.imshow(predicted_image)
        plt.show()
        
        preview_again = input("Would you like to preview prediciton results on another random image? [y/n]: ")
        
def save_model(model):
    save_model = input("\nWould you like to save the model? (recommended) [y/n]: ")
        
    if save_model == "y":
        save_timestamp = time.strftime('%Y-%m-%d_%H%M', time.localtime())
        print("Would you like to save the model as in the HD5 or the SavedModel format? \nHD5 results in a single file with the extension .h5, SavedModel results in a nested folder directory with model asssets")
        print("Hint: More info on the difference between the two formats can be found here: https://www.tensorflow.org/tutorials/keras/save_and_load#save_the_entire_model")
        model_format = int(input("Input 1 for HD5 or 2 for SavedModel:"))

        if model_format == 1:
            print("Saving model in the HD5 format...")
            model_name = f'satellite segmentation full_{save_timestamp}.h5'
            model.save(model_name)
        elif model_format == 2:
            print("Saving entire model in the SavedModel format...")
            model_name = f'satellite segmentation full_{save_timestamp}'
            model.save(model_name)

        print("\nTraining process complete! Model has been trained and saved!")
    elif save_model == "n":
        print("\nTraining process complete! Model has been trained but NOT saved.")
        print("(hint: to save model manually you can do so with model.save()")
    
    
def execute_model(labels, image_dataset):

    model, X_train, X_test, y_train, y_test = create_deep_learning_model(labels, image_dataset)
    
    time.sleep(3)
    
    epochs, batch_size = customizable_training_params()
    
    train_ready = input("\nAre you ready to train the model? NOTE: This may take between 20 and 60 minutes, depending on your machine. [y/n]:")
    
    if train_ready == "y":
        print("\n----Training model. Please do not close this window.----")
        model_history = model.fit(X_train, y_train,
                                  batch_size=batch_size,
                                  verbose=1,
                                  epochs=epochs,
                                  validation_data=(X_test, y_test),
                                  shuffle=False)
        
        print(f"\nTraining complete!")
        
        test_model(model, X_test, y_test)
        
        save_model(model)
    
        
        return model_history, model
    if train_ready ==  "n":
        print("Terminating process.")

    
        
    
    
 
