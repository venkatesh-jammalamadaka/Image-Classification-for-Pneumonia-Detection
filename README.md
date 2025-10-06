Pneumonia Detection using Deep Learning

A comprehensive binary image classification project for detecting pneumonia from chest X-ray images using Convolutional Neural Networks (CNNs) and Transfer Learning techniques.

Project Overview:

This project implements and compares multiple deep learning architectures to automatically detect pneumonia from pediatric chest X-rays. The models assist radiologists in faster and more accurate diagnosis by classifying X-ray images as either Normal or Pneumonia.
Dataset

Source: Chest X-Ray Images (Pneumonia) from Kaggle
Total Images: 5,856 chest X-rays
Classes: Binary (Normal vs Pneumonia)
Patient Age: 1-5 years old

Distribution:

Training: 5,216 images (3,875 pneumonia, 1,341 normal)
Testing: 624 images (390 pneumonia, 234 normal)
Validation: 16 images (8 pneumonia, 8 normal)



Key Features

Multiple CNN Architectures: Custom CNN, DenseNet121, VGG16, ResNet50, InceptionV3
Transfer Learning: Two-phase training (freeze → fine-tune) for optimal performance
Class Imbalance Handling: Weighted loss function to address 3:1 pneumonia-to-normal ratio
Data Augmentation: Rotation, shifting, shearing, zooming, and flipping to improve generalization
Medical-Appropriate Metrics: Sensitivity, specificity, precision, recall, AUC-ROC
Best Practices: Proper train/test isolation, no data leakage, early stopping, learning rate scheduling


Key Implementation Details:

1. Data Preprocessing

Images resized to 224×224 pixels
Pixel values rescaled to [0, 1] range
Training data augmented with geometric transformations


2. Transfer Learning Strategy
Phase 1 (Feature Extraction):

Freeze pre-trained base model
Train only custom top layers
Learning rate: 0.001
Epochs: 10

Phase 2 (Fine-Tuning):

Unfreeze top 20-30 layers of base model
Train with very low learning rate: 0.00001
Epochs: 10

3. Class Imbalance Handling
pythonweight_normal = 1.94
weight_pneumonia = 0.67

4. Callbacks

Used 'Early Stopping' and 'Learning Rate Reduction'


Results Visualization
The project generates:

Training history plots (loss, accuracy, AUC, recall over epochs)
ROC curves for each model
Confusion matrices
Comparative bar charts across all metrics

Results:
Winner: DenseNet121 achieved the best balance of sensitivity and specificity with the highest AUC-ROC score.

Sensitivity (Recall): 94.10% means model catches 94% of pneumonia cases
Specificity: 89.74% means model correctly identifies 90% of normal cases
23 False Negatives: Missed pneumonia cases (most critical error)
24 False Positives: Unnecessary follow-ups (less critical)

Improvements Over Baseline

Fixed data leakage from original implementation
Proper two-phase transfer learning (20% accuracy improvement)
Increased image resolution (180×180 → 224×224)
Consistent architecture for fair model comparison
Medical-appropriate evaluation metrics

Future Enhancements

 Implement Grad-CAM for model interpretability
 Ensemble multiple models for better performance
 External validation on different hospital datasets
 Adjust decision threshold for higher sensitivity
 Deploy as REST API for clinical integration
 Add confidence intervals for predictions

References

Original Dataset: Kermany et al., 2018
DenseNet: Huang et al., 2017
VGG16: Simonyan & Zisserman, 2014
ResNet: He et al., 2015
InceptionV3: Szegedy et al., 2015


Acknowledgments:

Dataset provided by Guangzhou Women and Children's Medical Center
Kaggle for hosting the dataset and providing free GPU resources
TensorFlow and Keras teams for excellent deep learning frameworks