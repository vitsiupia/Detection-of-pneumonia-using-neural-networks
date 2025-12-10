# Detection-of-pneumonia-using-neural-networks
🫁 Pneumonia detection from Chest X-Rays using a custom ResNet architecture in PyTorch. Accuracy: 83%.

This project, completed as part of a course assignment, aims to classify chest X-ray images into two categories: NORMAL (healthy) and PNEUMONIA (pneumonia). It is based on a custom ResNet architecture, implemented from scratch and adapted to the specific characteristics of medical imaging data. The project also examines the issue of overfitting and uses a checkpointing mechanism to save the best-performing version of the model.

Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data)  from Kaggle.

Technologies:
- Python 3
- PyTorch (model building and training)
- Torchvision (image transformations)
- Matplotlib / Seaborn (result visualization)
- Google Colab (GPU-enabled environment)

Model Architecture
A custom implementation of a ResNet-like network was used, featuring a key modification:
Single-channel Input (Grayscale): Instead of standard RGB, the first convolutional layer was adapted to grayscale X-ray images (in_channels=1), optimizing the training process.
Output: 2 classes (Binary Classification).

Results:
The model achieved 83% accuracy on the validation set.
- Overfitting was detected after the 2nd epoch.
- Early Stopping was applied (selecting the best weights from epoch 2).
- A Confusion Matrix was generated to analyze errors such as false negatives.
