# Cat Dog Recognizer (CNN)

This project is an **image classification** system that distinguishes **cats** and **dogs**
using a **Convolutional Neural Network (CNN)**.  
It includes training, testing, and visualizing predictions.

---

## Project Structure

- `load_mnist.py`  
  (Filename kept for consistency, but used for cat/dog images)
  - Loads images from folders.
  - `load_images_from_folder(path)` → returns `(num_images, 256, 256, 3)`
  - Images are loaded in RGB format and resized to `256×256`.

- `model.py`  
  - Defines the CNN model with:
    - Conv2D + MaxPooling2D  
    - GlobalAveragePooling2D  
    - Dense layers  
    - Sigmoid output (binary classification)  
  - `create_model()` returns a Keras model.

- `train.py`  
  - Loads cat and dog training images.
  - Creates labels:
    - Cat → `0`
    - Dog → `1`
  - Combines and shuffles training data.
  - Creates model via `model.py`, compiles it, loads existing weights if available.
  - Trains the model and saves weights (`my_cat_dog_model.weights.h5`).

- `test.py`  
  - Loads test images.
  - Loads trained model and weights.
  - Predicts results using sigmoid output:
    - `> 0.5` → Dog
    - `≤ 0.5` → Cat
  - Prints **correct / total predictions** and **test accuracy**.
  - Can display misclassified images with predicted and true labels.

---

## Dataset

- Cat and Dog image dataset (binary classification).
- Images are organized in folders:
  - Cat images
  - Dog images
- Images are resized to `256×256` and normalized before training.

**Note:** You do not need to label images manually.

---

## Optional: Use a Virtual Environment & How to Run

```bash
# 1. Create and activate virtual environment (optional but recommended)
python -m venv venv
.\venv\Scripts\activate

# 2. Install required packages
pip install tensorflow numpy matplotlib

# 3. Train the model
python train.py

# 4. Test the model
python test.py
