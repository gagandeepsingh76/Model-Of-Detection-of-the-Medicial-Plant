# Plant Image Classification System

This system uses deep learning to classify medicinal and edible plants from images. It implements four different CNN-based models and provides functionality for both training and prediction.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your dataset is organized in the following structure:
```
train_dir/
    plant_class_1/
        image1.jpg
        image2.jpg
        ...
    plant_class_2/
        image1.jpg
        image2.jpg
        ...
    ...

val_dir/
    plant_class_1/
        image1.jpg
        image2.jpg
        ...
    plant_class_2/
        image1.jpg
        image2.jpg
        ...
    ...
```

## Training Models

To train all four models, simply run:
```bash
python plant_classifier.py
```

This will:
1. Train four different models:
   - Custom CNN (m1.h5)
   - MobileNetV2-based model (m2.h5)
   - EfficientNetB0-based model (m3.h5)
   - ResNet50-based model (m4.h5)
2. Save the best model weights for each architecture
3. Generate evaluation metrics and visualizations
4. Create a class mapping file (labels.txt)

## Making Predictions

To classify a new plant image, use the `predict_image` function:

```python
from plant_classifier import predict_image

result = predict_image(
    image_path="path/to/your/image.jpg",
    model_path="m1.h5"  # or m2.h5, m3.h5, m4.h5
)

print(f"Predicted class: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
print("Similar images:", result['similar_images'])
```

## Output Files

The system generates several output files:
- `m1.h5`, `m2.h5`, `m3.h5`, `m4.h5`: Trained model weights
- `labels.txt`: Class-to-index mapping
- `confusion_matrix_*.png`: Confusion matrices for each model
- `training_history_*.png`: Training/validation accuracy and loss curves

## Model Architectures

1. **Custom CNN (m1.h5)**
   - Simple CNN with Conv2D, MaxPooling2D, and Dense layers
   - Good for smaller datasets and faster training

2. **MobileNetV2 (m2.h5)**
   - Lightweight model suitable for mobile devices
   - Good balance between accuracy and speed

3. **EfficientNetB0 (m3.h5)**
   - State-of-the-art architecture
   - Excellent accuracy with reasonable computational cost

4. **ResNet50 (m4.h5)**
   - Deep residual network
   - Strong performance on complex classification tasks

## Training Parameters

- Image size: 224x224 pixels
- Batch size: 32
- Maximum epochs: 30
- Early stopping patience: 5 epochs
- Learning rate: 0.001 with reduction on plateau
- Data augmentation: rotation, flip, zoom, shift, brightness adjustment 

def plot_training_history(history, model_name):
    plt.figure(figsize=(12, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'training_history_{model_name}.png')
    plt.close() 

for model_name, model_creator in models_to_train.items():
    print(f"\nTraining {model_name}...")
    model = model_creator(num_classes)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    history = train_model(model, model_name, train_generator, val_generator)
    evaluate_model(model, val_generator, list(class_mapping.values()))
    plot_training_history(history, model_name) 