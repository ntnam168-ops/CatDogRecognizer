import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from PIL import Image

cat_train = "D:/AI project/CatDogRecognizer/CatDogRecognizer/data/train/cats"
dog_train = "D:/AI project/CatDogRecognizer/CatDogRecognizer/data/train/dogs"
cat_test = "D:/AI project/CatDogRecognizer/CatDogRecognizer/data/test/cats"
dog_test = "D:/AI project/CatDogRecognizer/CatDogRecognizer/data/test/dogs"
image = []
label = []

def load_images_from_folder(folder):
    image = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path).convert("RGB")
        img = img.resize((256, 256))
        img_array = np.array(img)
        image.append(img_array)
    return np.array(image)

# dog_train_img = load_images_from_folder(dog_train)
# cat_train_img = load_images_from_folder(cat_train)
# dog_test_img = load_images_from_folder(dog_test)
# cat_test_img = load_images_from_folder(cat_test)

# print("Dog train images:", dog_train_img.shape)
# print("Cat train images:", cat_train_img.shape)
# print("Dog test images:", dog_test_img.shape)
# print("Cat test images:", cat_test_img.shape) 