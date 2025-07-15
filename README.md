# Pneumonia-detection
This project shows how to build a practical pneumonia detector using transfer learning, smart data augmentation, and careful evaluation to get robust results on limited data

## Methodology
The goal of this project is to build a binary image classifier that can detect pneumonia in chest X-ray images.

Key steps in the methodology:

### a) Transfer Learning
Instead of training a CNN from scratch (which requires huge data), I reuse VGG16, a popular convolutional neural network pre-trained on ImageNet.

I remove its top fully connected layers (include_top=False).

I freeze its convolutional layers initially so they act as a fixed feature extractor for X-ray images.

### b) Custom Classification Head
After the VGG16 base, I add:

Flatten layer to turn feature maps into a vector.

Dense layer (256 units) with ReLU activation for learning new patterns.

Dropout layer to reduce overfitting.

Dense output layer with sigmoid activation to output probability (0: healthy, 1: pneumonia).

### c) Data Augmentation
To prevent overfitting on a small dataset, I artificially generate variations of images:

Random rotations, flips, zooms, shifts.

This helps the model generalize better to unseen X-rays.

### d) Training Setup
Loss function: Binary cross-entropy, standard for binary classification.

Optimizer: RMSprop with a low learning rate (1e-4) to carefully adjust pre-trained weights.

Early stopping: Monitors validation loss, stops training if performance stops improving.

### e) Evaluation
After training, the model is evaluated on an unseen test dataset.

I analyze:

Accuracy and loss curves.

Confusion matrix.

Classification report (precision, recall, F1-score)


## Data
The dataset consists of chest X-ray images categorized into two classes:

PNEUMONIA: X-rays showing signs of pneumonia.

NORMAL: X-rays of healthy lungs.

Data folders:

train/ — used to train the model.

val/ — used to validate during training to monitor overfitting.

test/ — unseen data to evaluate final performance.

## Results

Key results:

- Training accuracy improves steadily, showing that the model learns the training data well.

- Validation accuracy stabilizes, showing that the model generalizes to unseen validation data — a good sign when data augmentation is used.

- Loss plots show that training loss decreases while validation loss stays relatively flat — meaning overfitting is reduced but may still exist if the gap widens too much.

- Test accuracy (in your runs) was ~78% — this shows the model is significantly better than random guessing (50% for binary), but there’s room to improve with more data and fine-tuning.

- Confusion matrix shows how many healthy cases are misclassified as pneumonia and vice versa.

- Classification report gives precision and recall for both classes — useful in healthcare, where false negatives (missing pneumonia) can be critical.


## ⚙️ Requirements

- Python 3.x
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn



