import cv2
import numpy as np
import os
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import pickle

class_0_path = "D:/Computer_Vision_Fifa/dataset/false"
class_1_path = "D:/Computer_Vision_Fifa/dataset/true"

def extract_hog_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (128, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = hog(gray, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    return features

features = []
image_labels = []
image_paths = []

for filename in os.listdir(class_0_path):
    if filename.endswith(('.jpg', '.png')):
        path = os.path.join(class_0_path, filename)
        feature_vector = extract_hog_features(path)
        if feature_vector is not None:
            features.append(feature_vector)
            image_labels.append(0)
            image_paths.append(path)

count = 0
for filename in os.listdir(class_1_path):
    if filename.endswith(('.jpg', '.png')):
        path = os.path.join(class_1_path, filename)
        feature_vector = extract_hog_features(path)
        if feature_vector is not None:
       
            features.append(feature_vector)
            image_labels.append(1)
            image_paths.append(path)

features = np.array(features)
labels = np.array(image_labels)

X_train, X_test, y_train, y_test, train_paths, test_paths = train_test_split(
    features, labels, image_paths, test_size=0.2, random_state=42
)

svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Calculate precision, recall, and F1 score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nPrecision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Display confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Display detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model
with open('svm_hog_model.pkl', 'wb') as f:
    pickle.dump(svm, f)

print("Results:")
for path, true_label, pred_label in zip(test_paths, y_test, y_pred):
    print(f"Image: {path}, Got: {true_label}, Expected: {pred_label}")
