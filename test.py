import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

class Config:
    IMG_SIZE = (224, 224)
    MODEL_PATH = r"C:\Users\PC\Desktop\visioncontest\food_comparator_model.keras"
    TEST_CSV = r"C:\Users\PC\Desktop\visioncontest\Test Set Samples\test.csv"
    TEST_IMAGES_FOLDER = r"C:\Users\PC\Desktop\visioncontest\Test Set Samples\Test Images"

def load_and_process_image(img_path):
    try:
        full_path = os.path.join(Config.TEST_IMAGES_FOLDER, img_path)
        if os.path.exists(full_path):
            img = load_img(full_path, target_size=Config.IMG_SIZE)
            img = img_to_array(img) / 255.0
            return img
        else:
            print(f"Image not found: {full_path}")
            return np.zeros((*Config.IMG_SIZE, 3))
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return np.zeros((*Config.IMG_SIZE, 3))

def predict_image_pair(model, img1_path, img2_path):
    processed_img1 = load_and_process_image(img1_path)
    processed_img2 = load_and_process_image(img2_path)
    
    score1 = model.predict(np.expand_dims(processed_img1, axis=0))[0][0]
    score2 = model.predict(np.expand_dims(processed_img2, axis=0))[0][0]
    
    return 1 if score1 > score2 else 2

def main():
    # Load the trained model
    try:
        model = load_model(Config.MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load test CSV
    try:
        test_df = pd.read_csv(Config.TEST_CSV)
    except Exception as e:
        print(f"Error loading test CSV: {e}")
        return
    
    # Predict and track results
    results = []
    correct_predictions = 0
    total_predictions = len(test_df)
    
    for _, row in test_df.iterrows():
        try:
            # Predict winner
            predicted_winner = predict_image_pair(model, row['Image 1'], row['Image 2'])
            results.append(predicted_winner)
            
            # Check accuracy only if original ground truth exists
            if row['Winner'] != 0:
                if predicted_winner == row['Winner']:
                    correct_predictions += 1
        
        except Exception as e:
            print(f"Error predicting pair: {e}")
            results.append(0)
    
    # Calculate accuracy
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    
    # Update test CSV with predictions
    test_df['Predicted_Winner'] = results
    test_df.to_csv(Config.TEST_CSV, index=False)
    
    # Detailed Reporting
    print("\n--- Prediction Results ---")
    print(f"Total Predictions: {total_predictions}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Breakdown by original ground truth
    print("\nDetailed Breakdown:")
    print(test_df[['Image 1', 'Image 2', 'Winner', 'Predicted_Winner']])

if __name__ == "__main__":
    main()