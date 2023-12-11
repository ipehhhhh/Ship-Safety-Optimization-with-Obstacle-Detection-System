import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels

def read_annotations_csv(annotations_file):
    data = []

    with open(annotations_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line:  # Skip empty lines
                parts = line.split(' ')
                filename = parts[0]
                class_label = None

                if len(parts) > 1:
                    class_label = parts[1].split(',')[4] if ',' in parts[1] else None

                data.append({'filename': filename, 'class': class_label})

    return pd.DataFrame(data)

def read_classes_csv(classes_file):
    df = pd.read_csv(classes_file, header=None, names=['class'])
    le = LabelEncoder()
    df['class_label'] = le.fit_transform(df['class'])
    return df

def read_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            images.append({'filename': filename, 'image': img})
    return images

def classify_images(images, annotations_df, classes_df):
    data = []
    target = []

    for image_info in images:
        filename = image_info['filename']
        class_label = annotations_df.loc[annotations_df['filename'] == filename, 'class'].values[0]

        if class_label is not None:
            target.append(class_label)
            data.append(image_info['image'].flatten())

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    report = classification_report(y_test, y_pred)
    print(f'Classification Report:\n{report}')

    # Confusion Matrix Visualization
    plot_confusion_matrix(y_test, y_pred, classes=classes_df['class'].values)

def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    classes = unique_labels(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    folder_path = r'C:\Users\syari\Downloads\AutoLabelling.v1i.tensorflow\validYOLO'
    annotations_file = r'C:\Users\syari\Downloads\AutoLabelling.v1i.tensorflow\validYOLO\_annotations.txt'
    classes_file = r'C:\Users\syari\Downloads\AutoLabelling.v1i.tensorflow\validYOLO\_classes.txt'

    annotations_df = read_annotations_csv(annotations_file)
    classes_df = read_classes_csv(classes_file)
    images = read_images_from_folder(folder_path)

    classify_images(images, annotations_df, classes_df)
