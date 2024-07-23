
## Fine-tuning a Pre-trained Image Classifier for Flower Classification

### Step 1: Data Preparation

1. **Data Loading and Preprocessing**:
   - `X` holds the image data.
   - `Z` holds the corresponding labels (e.g., if the image contains a daisy flower, it is labeled as daisy).
   - `img_size` is set to 150, indicating that all images are resized to 150x150 pixels.

2. **Functions**:
   - **Assign Label**: Takes an image and a flower type as input and returns the flower type as the label.
   - **Load Images**: Takes the flower type and its directory as input, iterates over all images in that directory, assigns labels using `assign_label`, reads and resizes each image, and appends the image data and corresponding label to lists `X` and `Z`.

3. **Dataset Overview**:
   - Total records: 4,317 entries (flowers).
   - Label encoding converts categorical labels into numerical labels (e.g., Daisy → 0, Rose → 1, etc.).
   - One-hot encoding is used to prepare the labels for training a classification model.

4. **Dataset Split**:
   - Training dataset: 75%
   - Test dataset: 25%
   - Shapes:
     - `X_train`: (3237, 150, 150, 3)
     - `X_test`: (1080, 150, 150, 3)
     - `Y_train`: (3237, 5)
     - `Y_test`: (1080, 5)

### Step 2: Initial Model Performance Evaluation

- Loaded a pre-trained ResNet50 model.
- Customized the model for flower classification by adding custom layers on top of the pre-trained model.
- Freezed the pre-trained layers to prevent them from being updated.
- Compiled the model using the Adam optimizer, categorical cross-entropy loss function, and accuracy metric.
- Evaluated the performance of the model on the test set.
- Initial results: Accuracy of 16% with a test loss of 25.04.

### Step 3: Implementing Transfer Learning in TensorFlow

1. **Model Setup**:
   - Loaded a pre-trained ResNet50 model with weights pretrained on ImageNet data.
   - Freezed the pre-trained layers to retain their weights.
   - Added custom layers on top of the pre-trained ResNet50 model:
     - Global average pooling layer.
     - Two dense layers with ReLU and softmax activations.
   - Compiled the model using the Adam optimizer, categorical cross-entropy loss function, and accuracy metric.

2. **Training**:
   - Trained the model using the training data for 20 epochs with a batch size of 32.
   - Used validation data to monitor the model's performance during training.

3. **Evaluation**:
   - Achieved a test loss of 0.6485 and an accuracy of 88.05%.
   - Performance metrics:
     - Accuracy: 0.8806
     - Precision: 0.8813
     - Recall: 0.8806
     - F1-score: 0.8805

4. **Misclassification Analysis**:
   - Daisy: Correctly classified 173, misclassified 15 sunflowers, 5 tulips, 10 dandelions, 4 roses.
   - Sunflower: Correctly classified 227, misclassified 15 daisies, 3 tulips, 4 dandelions.
   - Tulip: Correctly classified 165, misclassified 3 daisies, 2 dandelions, 18 roses.
   - Dandelion: Correctly classified 183, misclassified 8 sunflowers, 3 tulips, 5 daisies, 2 roses.
   - Rose: Correctly classified 204, misclassified 6 sunflowers, 24 tulips, 2 dandelions.

### Step 4: Implementing Transfer Learning in PyTorch

1. **Data Handling**:
   - Used `TensorDataset` to create datasets directly from tensors without defining a custom dataset class.
   - Included additional preprocessing steps like image resizing.

2. **Model Setup and Training**:
   - Similar to the TensorFlow approach, a pre-trained ResNet50 model was used.
   - Achieved an accuracy of 83%.

### Comparison of TensorFlow and PyTorch

- **TensorFlow**:
  - Achieved higher accuracy (88.05%).
  - Utilized additional custom layers on top of the pre-trained ResNet50 model for better feature extraction and classification.

- **PyTorch**:
  - Achieved an accuracy of 83%.
  - Simplified data handling and preprocessing steps contributed to performance improvements.

- **Common Aspects**:
  - Both models froze the pre-trained layers to retain knowledge from the ImageNet dataset.
  - Used the Adam optimizer and categorical cross-entropy loss function for consistency in optimization and evaluation.

### Conclusion

- The TensorFlow model, with additional custom layers, outperformed the PyTorch model in terms of accuracy.
- Preprocessing steps such as image resizing and normalization contributed to improved performance in PyTorch.
- Overall, TensorFlow demonstrated better performance and efficacy for the task of flower classification compared to PyTorch.

License
This project is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0) License.

You are free to:

Share — copy and redistribute the material in any medium or format
Under the following terms:

Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
NonCommercial — You may not use the material for commercial purposes.
NoDerivatives — If you remix, transform, or build upon the material, you may not distribute the modified material.
For more details, see the license details.