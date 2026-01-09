import os 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from model import create_model
from load_mnist import load_images_from_folder, cat_test, dog_test

cat_test_img = load_images_from_folder(cat_test)
dog_test_img = load_images_from_folder(dog_test)

y_cat = np.zeros(len(cat_test_img))
y_dog = np.ones(len(dog_test_img))

x_test = np.concatenate((cat_test_img, dog_test_img), axis=0)
y_test = np.concatenate((y_cat, y_dog), axis=0)

indices = np.arange(len(x_test))
np.random.shuffle(indices)
x_test = x_test[indices]
y_test = y_test[indices]

model = create_model()
weights_file = "D:/AI project/CatDogRecognizer/my_cat_dog_model.weights.h5"
if os.path.exists(weights_file):
    model.load_weights(weights_file)
    print("Loaded existing weights.")

y_pred_prob = model.predict(x_test, batch_size=1)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

correct = np.sum(y_pred == y_test)
total = len(y_test)
accuracy = correct / total

print(f"Correct / Total: {correct} / {total}")
print(f"Test Accuracy: {accuracy:.4f}")

wrong_count = 0
for i in range(total):
    if y_pred[i] != y_test[i]:
        wrong_count += 1

        img = x_test[i]
        pred_label = "Dog" if y_pred[i] == 1 else "Cat"
        true_label = "Dog" if y_test[i] == 1 else "Cat"
        prob = y_pred_prob[i][0]

        print(f"[WRONG #{wrong_count}]  AI: {pred_label} | True: {true_label} | Prob: {prob:.3f}")

        plt.figure(figsize=(4,4))
        plt.imshow(img)
        plt.title(f"AI: {pred_label} ({prob:.2f}) | True: {true_label}")
        plt.axis("off")
        plt.show()

print("Total wrong predictions:", wrong_count)
