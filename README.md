# Cat Dog Recognizer (CNN)

This project is an image classification system that distinguishes **cats and dogs**
using a **Convolutional Neural Network (CNN)**.

---

## Project Structure

### load_mnist.py

- Loads images from folders.
- load_images_from_folder(path)
  → returns (num_images, 256, 256, 3)
- Images are loaded in RGB format and resized to 256×256.

Defined paths:
- cat_train, dog_train
- cat_test, dog_test

---

### model.py

Defines the CNN architecture:
- Conv2D + MaxPooling2D layers
- GlobalAveragePooling2D
- Dense layers
- Sigmoid output layer (binary classification)

Output meaning:
- 0 → Cat
- 1 → Dog

create_model() returns a Keras model.

---

### train.py

- Loads cat and dog training images.
- Creates labels:
  - Cat → 0
  - Dog → 1
- Combines and shuffles the dataset.
- Builds the model using model.py.
- Loads existing weights if available.
- Trains the model and saves weights to:

my_cat_dog_model.weights.h5

---

### test.py

- Loads test images (cats and dogs).
- Loads trained model and weights.
- Predicts results using sigmoid output:
  - > 0.5 → Dog
  - ≤ 0.5 → Cat
- Prints:
  - Correct predictions
  - Total samples
  - Test accuracy
- Can display misclassified images with:
  - Image
  - Predicted label
  - True label

---

## Dataset

Cat & Dog image dataset (real photographs).

Directory structure:
dataset/
 ├── train/
 │    ├── cat/
 │    └── dog/
 └── test/
      ├── cat/
      └── dog/

You do not need to draw cats or dogs yourself.

---

## Requirements

Install required packages:

pip install tensorflow numpy matplotlib

(Optional) Virtual environment:

python -m venv venv  
.\venv\Scripts\activate

---

## How to Run

Train the model:
python train.py

Test the model:
python test.py

---

## Notes

- Binary classification problem
- Uses sigmoid + binary_crossentropy
- Accuracy depends on dataset quality
- Large images (256×256) require more memory
