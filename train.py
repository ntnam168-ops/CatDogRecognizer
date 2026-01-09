import os 
import numpy as np
import tensorflow as tf
from model import create_model
from load_mnist import load_images_from_folder, cat_train, dog_train

cat_train_img = load_images_from_folder(cat_train)
dog_train_img = load_images_from_folder(dog_train)

y_cat = np.zeros(len(cat_train_img))
y_dog = np.ones(len(dog_train_img))

x_train = np.concatenate((cat_train_img, dog_train_img), axis=0)
y_train = np.concatenate((y_cat, y_dog), axis=0)

indices = np.arange(len(x_train))
np.random.shuffle(indices)
x_train = x_train[indices]
y_train = y_train[indices]

model = create_model()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

weights_file = "D:/AI project/CatDogRecognizer/my_cat_dog_model.weights.h5"
if os.path.exists(weights_file):
    model.load_weights(weights_file)
    print("Loaded existing weights.")

# for i in range(100):
#     print("TEST #", i+1)

#     num_samples = 32
#     indices = np.random.choice(len(x_train), size=num_samples, replace=False)

#     x_batch = x_train[indices]
#     y_batch = y_train[indices]

#     model.fit(x_batch, y_batch, epochs=10, batch_size=16)

model.fit(x_train, y_train, epochs=10, batch_size=8)

model.save_weights(weights_file)
print("Weights saved to my_cat_dog_model.weights.h5")

loss, acc = model.evaluate(x_train, y_train)
print("Training Accuracy:", acc)