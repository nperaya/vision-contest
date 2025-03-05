import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# Configuration
class Config:
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    
    # Paths (use relative paths)
    QUESTIONNAIRE_PATH = r"C:\Users\PC\Desktop\visioncontest\Questionair Images\Questionair Images"
    INSTAGRAM_PATH = r"C:\Users\PC\Desktop\visioncontest\Instagram Photos\Intragram Images [Original]"
    
    # CSV files
    QUESTIONNAIRE_CSV = "data_from_questionaire.csv"
    INSTAGRAM_CSV = "data_from_intragram.csv"
    
    # Model save path
    MODEL_SAVE_PATH = "food_comparator_model.keras"

def load_and_process_image(img_path, base_folder, has_subfolders=False):
    """
    Robust image loading with comprehensive error handling
    """
    try:
        if has_subfolders:
            for category in os.listdir(base_folder):
                category_path = os.path.join(base_folder, category)
                full_path = os.path.join(category_path, img_path)
                if os.path.exists(full_path):
                    img = load_img(full_path, target_size=Config.IMG_SIZE)
                    img = img_to_array(img) / 255.0
                    return img
        else:
            full_path = os.path.join(base_folder, img_path)
            if os.path.exists(full_path):
                img = load_img(full_path, target_size=Config.IMG_SIZE)
                img = img_to_array(img) / 255.0
                return img
        
        print(f"[WARNING] Image not found: {img_path}")
        return np.zeros((*Config.IMG_SIZE, 3))
    
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return np.zeros((*Config.IMG_SIZE, 3))

def prepare_dataset():
    """
    Prepare dataset from CSV files
    """
    X_images, y = [], []

    # Load Questionnaire data
    df_questionnaire = pd.read_csv(Config.QUESTIONNAIRE_CSV)
    for _, row in df_questionnaire.iterrows():
        img1 = load_and_process_image(row['Image 1'], Config.QUESTIONNAIRE_PATH)
        img2 = load_and_process_image(row['Image 2'], Config.QUESTIONNAIRE_PATH)

        if row['Winner'] == 1:
            X_images.extend([img1, img2])
            y.extend([1, 0])
        else:
            X_images.extend([img2, img1])
            y.extend([1, 0])

    # Load Instagram data
    df_instagram = pd.read_csv(Config.INSTAGRAM_CSV)
    for _, row in df_instagram.iterrows():
        img1 = load_and_process_image(row['Image 1'], Config.INSTAGRAM_PATH, has_subfolders=True)
        img2 = load_and_process_image(row['Image 2'], Config.INSTAGRAM_PATH, has_subfolders=True)

        if row['Winner'] == 1:
            X_images.extend([img1, img2])
            y.extend([1, 0])
        else:
            X_images.extend([img2, img1])
            y.extend([1, 0])

    return np.array(X_images), np.array(y)

def create_model():
    """
    Create transfer learning model with ResNet50
    """
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    
    # Compile with custom learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE)
    model.compile(optimizer=optimizer, 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    return model

def train_model():
    """
    Train the model with advanced callbacks
    """
    # Prepare data
    X, y = prepare_dataset()
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create model
    model = create_model()
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        Config.MODEL_SAVE_PATH, 
        save_best_only=True, 
        monitor='val_accuracy'
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2, 
        patience=5, 
        min_lr=1e-6
    )
    
    # Train
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val),
        batch_size=Config.BATCH_SIZE, 
        epochs=Config.EPOCHS,
        callbacks=[early_stopping, model_checkpoint, reduce_lr]
    )
    
    # Evaluate
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {val_accuracy*100:.2f}%")
    
    return model

def main():
    print("Starting Food Image Comparator Training...")
    trained_model = train_model()
    print("Training Completed. Model saved to:", Config.MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()