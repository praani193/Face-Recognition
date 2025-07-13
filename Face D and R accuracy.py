import os
import random
from deepface import DeepFace
import shutil

# Paths
full_dataset = "dataset1"
temp_train_dataset = "temp_dataset_train"
correct = 0
total = 0

# Step 1: Create temporary train dataset with first 150 images per person
if os.path.exists(temp_train_dataset):
    shutil.rmtree(temp_train_dataset)
os.makedirs(temp_train_dataset)

for person in os.listdir(full_dataset):
    person_path = os.path.join(full_dataset, person)
    if not os.path.isdir(person_path):
        continue

    images = sorted(os.listdir(person_path))  # sort to ensure order
    train_images = images[:150]

    person_train_path = os.path.join(temp_train_dataset, person)
    os.makedirs(person_train_path, exist_ok=True)

    for img in train_images:
        src = os.path.join(person_path, img)
        dst = os.path.join(person_train_path, img)
        shutil.copy(src, dst)

# Step 2: Randomly select 10 images from the first 150 for testing
for person in os.listdir(full_dataset):
    person_path = os.path.join(full_dataset, person)
    if not os.path.isdir(person_path):
        continue

    images = sorted(os.listdir(person_path))  # sort to ensure order
    train_images = images[:150]
    test_images = random.sample(train_images, 10)  # Randomly select 10 images

    for img in test_images:
        img_path = os.path.join(person_path, img)
        try:
            result = DeepFace.find(img_path=img_path, db_path=temp_train_dataset, model_name="Facenet512", enforce_detection=False)
            if len(result) > 0 and not result[0].empty:
                predicted_path = result[0]['identity'][0]
                predicted_name = predicted_path.split(os.sep)[-2]

                if predicted_name == person:
                    correct += 1
            total += 1
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# Step 3: Accuracy
accuracy = correct / total * 100 if total else 0
print(f"\nFace Recognition Accuracy: {accuracy:.2f}%")
