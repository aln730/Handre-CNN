import os
import numpy as np
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

image_size = (28, 28)
num_classes = 27
data_dir = 'your_dataset_directory_here'

def load_images_from_directory(directory):
    images = []
    labels = []
    
    for label, person_folder in enumerate(os.listdir(directory)):
        person_folder_path = os.path.join(directory, person_folder)
        
        if os.path.isdir(person_folder_path):
            for image_name in os.listdir(person_folder_path):
                if image_name.endswith('.png'):
                    image_path = os.path.join(person_folder_path, image_name)
                    
                    img = load_img(image_path, target_size=image_size, color_mode='grayscale')
                    img_array = img_to_array(img)
                    img_array = img_array / 255.0
                    
                    images.append(img_array)
                    labels.append(label)
                    
    return np.array(images), np.array(labels)

X, y = load_images_from_directory(data_dir)
X = X.reshape(-1, 28, 28, 1)
y = to_categorical(y, num_classes=num_classes)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_model()
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
model.save('cnn_model.h5')

def predict_image(model, image_path):
    img = load_img(image_path, target_size=image_size, color_mode='grayscale')
    img_array = img_to_array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    
    predictions = model.predict(img_array)
    predicted_label = np.argmax(predictions)
    
    return predicted_label, predictions[0][predicted_label]
