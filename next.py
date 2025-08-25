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

# Model 5: Swin Transformer (Tiny)
def create_swin_transformer(num_classes):
    # Install and import swin transformer if not available
    try:
        import swin_transformer
    except ImportError:
        print("Swin Transformer not available, using a simplified implementation")
        # Simplified Swin Transformer-like architecture
        model = models.Sequential([
            layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
            layers.Conv2D(96, (4, 4), strides=4, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            
            # Simplified transformer blocks
            layers.Conv2D(192, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(384, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(768, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.GlobalAveragePooling2D(),
            
            layers.Dense(1024, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model
    
    # If swin_transformer is available, use the actual implementation
    base_model = swin_transformer.SwinTransformerTiny224(
        include_top=False,
        weights='imagenet21k_1k_224',
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

# Model 6: EfficientNetV2 (Small)
def create_efficientnetv2(num_classes):
    base_model = applications.EfficientNetV2S(
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

# Model 7: RegNetY (RegNetY-16GF equivalent)
def create_regnet(num_classes):
    base_model = applications.RegNetY160(
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

# Model 8: CoAtNet (CoAtNet-1 equivalent)
def create_coatnet(num_classes):
    # Since CoAtNet is not directly available in TensorFlow, we'll create a simplified version
    # that mimics the CoAtNet architecture with Convolution and Attention blocks
    
    def conv_block(filters, kernel_size=3, strides=1):
        return models.Sequential([
            layers.Conv2D(filters, kernel_size, strides=strides, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
    
    def attention_block(filters):
        return models.Sequential([
            layers.MultiHeadAttention(num_heads=8, key_dim=filters//8),
            layers.Add(),
            layers.LayerNormalization(),
            layers.Dense(filters, activation='relu'),
            layers.Dense(filters),
            layers.Add(),
            layers.LayerNormalization()
        ])
    
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Initial convolution
    x = conv_block(64, 7, 2)(inputs)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # CoAtNet blocks (simplified)
    # Block 1: Conv
    x = conv_block(96)(x)
    x = conv_block(96)(x)
    x = layers.MaxPooling2D(2)(x)
    
    # Block 2: Conv + Attention
    x = conv_block(192)(x)
    x = conv_block(192)(x)
    # Reshape for attention
    batch_size = tf.shape(x)[0]
    height, width, channels = x.shape[1], x.shape[2], x.shape[3]
    x_reshaped = layers.Reshape((height * width, channels))(x)
    x_reshaped = attention_block(channels)(x_reshaped, x_reshaped)
    x = layers.Reshape((height, width, channels))(x_reshaped)
    x = layers.MaxPooling2D(2)(x)
    
    # Block 3: Attention
    x = conv_block(384)(x)
    batch_size = tf.shape(x)[0]
    height, width, channels = x.shape[1], x.shape[2], x.shape[3]
    x_reshaped = layers.Reshape((height * width, channels))(x)
    x_reshaped = attention_block(channels)(x_reshaped, x_reshaped)
    x = layers.Reshape((height, width, channels))(x_reshaped)
    x = layers.MaxPooling2D(2)(x)
    
    # Block 4: Attention
    x = conv_block(768)(x)
    batch_size = tf.shape(x)[0]
    height, width, channels = x.shape[1], x.shape[2], x.shape[3]
    x_reshaped = layers.Reshape((height * width, channels))(x)
    x_reshaped = attention_block(channels)(x_reshaped, x_reshaped)
    x = layers.Reshape((height, width, channels))(x_reshaped)
    
    # Global pooling and classification
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# Training function
def train_model(model, model_name, train_generator, val_generator):
    if model_name in ['m7', 'm8']:
        # For m7 and m8, only use model checkpointing and learning rate reduction
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
def evaluate_model(model, val_generator, class_names, model_name):
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
    
    # Set x-axis limits and ticks for models m7 and m8 to show all 30 epochs
    if model_name in ['m7', 'm8']:
        plt.xlim(1, 30)
        plt.xticks(range(1, 31, 5))  # Show ticks every 5 epochs
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['loss'], label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Set x-axis limits and ticks for models m7 and m8 to show all 30 epochs
    if model_name in ['m7', 'm8']:
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
        'm5': create_swin_transformer,
        'm6': create_efficientnetv2,
        'm7': create_regnet,
        'm8': create_coatnet
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
        evaluate_model(model, val_generator, list(class_mapping.values()), model_name)
        plot_training_history(history, model_name) 