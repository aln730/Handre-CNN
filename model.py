
import os
import numpy as np
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

# Parameters
image_size = (28, 28)  # Resize all images to 28x28
num_classes = 27  # Number of people 

# Function to load and preprocess data
def load_images_from_directory(directory):
    images = []
    labels = []
    
    for label, person_folder in enumerate(os.listdir(directory)):
        person_folder_path = os.path.join(directory, person_folder)
        
        if os.path.isdir(person_folder_path):
            for image_name in os.listdir(person_folder_path):
                if image_name.endswith('.png'):  # Assuming PNG images
                    image_path = os.path.join(person_folder_path, image_name)
                    
                    img = load_img(image_path, target_size=image_size, color_mode='grayscale')
                    img_array = img_to_array(img)  # Convert image to array
                    


                    img_array = img_array / 255.0
                    
                    images.append(img_array)
                    labels.append(label)
                    
    return np.array(images), np.array(labels)

