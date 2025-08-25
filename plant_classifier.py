import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import json
from pathlib import Path

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001

# Paths
TRAIN_DIR = r"C:\Users\hp\Desktop\models\test-20240602T045527Z-001"
VAL_DIR = r"C:\Users\hp\Desktop\models\validation-20240602T045418Z-001"

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess data
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb'
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb'
)

# Save class mapping
class_mapping = {v: k for k, v in train_generator.class_indices.items()}
with open('labels.txt', 'w') as f:
    json.dump(class_mapping, f, indent=4)

# Model 1: Custom CNN
def create_custom_cnn(num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Model 2: MobileNetV2
def create_mobilenetv2(num_classes):
    base_model = applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Model 3: EfficientNetB0
def create_efficientnet(num_classes):
    base_model = applications.EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Model 4: ResNet50
def create_resnet50(num_classes):
    base_model = applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Training function
def train_model(model, model_name, train_generator, val_generator):
    if model_name in ['m3', 'm4']:
        # For m3 and m4, only use model checkpointing and learning rate reduction
        callbacks = [
            ModelCheckpoint(f'{model_name}.h5', monitor='val_accuracy', save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
        ]
    else:
        # For other models, use all callbacks including early stopping
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
            ModelCheckpoint(f'{model_name}.h5', monitor='val_accuracy', save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
        ]
    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks
    )
    
    return history

# Evaluation function
def evaluate_model(model, val_generator, class_names):
    # Get predictions
    predictions = model.predict(val_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_generator.classes
    
    # Calculate metrics
    accuracy = np.mean(y_pred == y_true)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{model_name}.png')
    plt.close()
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    return accuracy

# Plot training history
def plot_training_history(history, model_name):
    plt.figure(figsize=(12, 4))
    
    # Get the number of epochs actually trained
    epochs = range(1, len(history.history['accuracy']) + 1)
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['accuracy'], label='Training Accuracy')
    plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Set x-axis limits and ticks for models m3 and m4 to show all 30 epochs
    if model_name in ['m3', 'm4']:
        plt.xlim(1, 30)
        plt.xticks(range(1, 31, 5))  # Show ticks every 5 epochs
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['loss'], label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Set x-axis limits and ticks for models m3 and m4 to show all 30 epochs
    if model_name in ['m3', 'm4']:
        plt.xlim(1, 30)
        plt.xticks(range(1, 31, 5))  # Show ticks every 5 epochs
    
    plt.tight_layout()
    plt.savefig(f'training_history_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Prediction function
def predict_image(image_path, model_path):
    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(
        image_path,
        target_size=(IMG_SIZE, IMG_SIZE)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Get prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    # Get class name
    with open('labels.txt', 'r') as f:
        class_mapping = json.load(f)
    predicted_class_name = class_mapping[str(predicted_class)]
    
    # Find similar images
    class_dir = os.path.join(TRAIN_DIR, predicted_class_name)
    similar_images = []
    if os.path.exists(class_dir):
        for img_file in os.listdir(class_dir)[:5]:  # Get top 5 similar images
            similar_images.append(os.path.join(class_dir, img_file))
    
    return {
        'predicted_class': predicted_class_name,
        'confidence': float(confidence),
        'similar_images': similar_images
    }

# Main execution
if __name__ == "__main__":
    num_classes = len(train_generator.class_indices)
    
    # Train and evaluate all models
    models_to_train = {
        'm1': create_custom_cnn,
        'm2': create_mobilenetv2,
        'm3': create_efficientnet,
        'm4': create_resnet50
    }
    
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