import cv2
import numpy as np
import os
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Paths to class folders
class_0_path = 'false_class'
class_1_path = 'true_class'

# Function to extract HOG descriptors from an image
def extract_hog_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (128, 256))  # Resize to fixed size
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Extract HOG features
    features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    return features

# Load images and extract HOG features
features = []
image_labels = []
image_paths = []

# Process class 0
for filename in os.listdir(class_0_path):
    if filename.endswith(('.jpg', '.png')):
        path = os.path.join(class_0_path, filename)
        feature_vector = extract_hog_features(path)
        if feature_vector is not None:
            features.append(feature_vector)
            image_labels.append(0)
            image_paths.append(path)

# Process class 1
count = 0
for filename in os.listdir(class_1_path):
    if filename.endswith(('.jpg', '.png')):
        path = os.path.join(class_1_path, filename)
        feature_vector = extract_hog_features(path)
        if feature_vector is not None:
       
            features.append(feature_vector)
            image_labels.append(1)
            image_paths.append(path)

# Convert to NumPy arrays
features = np.array(features)
labels = np.array(image_labels)

# Split into train/test sets
X_train, X_test, y_train, y_test, train_paths, test_paths = train_test_split(
    features, labels, image_paths, test_size=0.2, random_state=42
)

# Train SVM classifier
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)

# Evaluate on test set
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the model
with open('svm_hog_model.pkl', 'wb') as f:
    pickle.dump(svm, f)

# Print test results
print("\nTest Image Classifications:")
for path, true_label, pred_label in zip(test_paths, y_test, y_pred):
    print(f"Image: {path}, True Label: {true_label}, Predicted Label: {pred_label}")
