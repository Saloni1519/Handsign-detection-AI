import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from google.colab import drive
# Import the drive module from google.colab
from google.colab import drive

# Mount Google Drive (use this simple path)
drive.mount('/content/drive')
!mkdir -p hand_signs/train
!mkdir -p hand_signs/test
def collect_data():
    """
    Function to collect hand sign images using webcam
    Usage: Uncomment and run this function to collect data
    """
    # Categories for hand signs (example: 0-5 numbers)
    categories = ['zero', 'one', 'two', 'three', 'four', 'five']

    # Create directories for each category
    for category in categories:
        os.makedirs(f'hand_signs/train/{category}', exist_ok=True)

    # Capture images from webcam
    cap = cv2.VideoCapture(0)

    for category in categories:
        print(f"Collecting data for {category}. Press 'q' to move to next category.")
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Define ROI (Region of Interest) for hand
            roi = frame[100:400, 100:400]

            # Display ROI
            cv2.imshow("ROI", roi)

            # Save image every 5 frames
            if count % 5 == 0:
                cv2.imwrite(f'hand_signs/train/{category}/{count}.jpg', roi)
                print(f"Saved {count} images for {category}")

            count += 1

            # Press 'q' to move to next category
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
def load_data(data_dir, img_size=(64, 64)):
    """
    Load and preprocess images from directory
    """
    X = []
    y = []
    categories = os.listdir(data_dir)

    for idx, category in enumerate(categories):
        path = os.path.join(data_dir, category)
        if not os.path.isdir(path):
            continue

        print(f"Loading images from category: {category}")
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            try:
                # Read image
                img = cv2.imread(img_path)
                # Convert to RGB (OpenCV uses BGR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Resize
                img = cv2.resize(img, img_size)
                # Normalize pixel values
                img = img / 255.0

                X.append(img)
                y.append(idx)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    return np.array(X), np.array(y)
def prepare_data(X, y, test_size=0.2):
    """
    Split data into train and test sets and one-hot encode labels
    """
    # One-hot encode labels
    y = to_categorical(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test

# Data augmentation
def create_data_generator():
    """
    Create data generator for training with augmentation
    """
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    return train_datagen
def build_model(input_shape, num_classes):
    """
    Build CNN model for hand sign detection
    """
    model = Sequential([
        # First convolutional layer
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),

        # Second convolutional layer
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Third convolutional layer
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Flatten layer
        Flatten(),

        # Dense layers
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
def train_model(model, X_train, y_train, X_test, y_test, batch_size=32, epochs=15):
    """
    Train the model with data augmentation
    """
    # Create data generator
    train_datagen = create_data_generator()

    # Train the model
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test)
    )

    return history
def evaluate_model(model, history, X_test, y_test):
    """
    Evaluate model performance
    """
    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")

    # Plot training history
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def save_model(model, path='/content/drive/MyDrive/DATASET'):
    """
    Save the trained model
    """
    model.save(path)
    print(f"Model saved to {path}")
def predict_live():
    """
    Real-time prediction using webcam
    """
    # Load model
    model = tf.keras.models.load_model('/content/drive/MyDrive/DATASET')

    # Categories - replace with your actual class names
    categories = ['zero', 'one', 'two', 'three', 'four', 'five']

    # Start webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Define ROI
        roi = frame[100:400, 100:400]

        # Draw rectangle around ROI
        cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)

        # Preprocess image
        img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # Make prediction
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)

        # Display result
        cv2.putText(frame, f"Prediction: {categories[predicted_class]}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show frame
        cv2.imshow('Hand Sign Detection', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
def main():
    """
    Main function to run the hand sign detection project
    """
    # Set dataset path - CHANGE THIS TO YOUR ACTUAL PATH
    dataset_path = '/content/drive/MyDrive/DATASET'

    # Option 1: Collect your own data (uncomment to use)
    # collect_data()

    # Option 2: Use existing dataset
    # Check if dataset directory exists
    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} does not exist.")
        print("Please create the directory and add your dataset, or collect data using collect_data() function.")
        return

    # Load data
    print("Loading data...")
    X, y = load_data(dataset_path)

    # Check if data was loaded successfully
    if len(X) == 0:
        print("No images were loaded. Please check your dataset path.")
        return

    print(f"Loaded {len(X)} images with shape {X[0].shape}")

    # Prepare data
    print("Preparing data...")
    X_train, X_test, y_train, y_test = prepare_data(X, y)

    # Build model
    print("Building model...")
    input_shape = X_train[0].shape  # (64, 64, 3)
    num_classes = y_train.shape[1]  # Number of classes
    model = build_model(input_shape, num_classes)

    # Display model summary
    model.summary()

    # Train model
    print("Training model...")
    history = train_model(model, X_train, y_train, X_test, y_test)

    # Evaluate model
    print("Evaluating model...")
    evaluate_model(model, history, X_test, y_test)

    # Save model
    save_model(model)

    print("Model training complete!")

    # Optional: Real-time prediction (uncomment to use)
    # print("Starting real-time prediction...")
    # predict_live()
if name == "main":
    main()
