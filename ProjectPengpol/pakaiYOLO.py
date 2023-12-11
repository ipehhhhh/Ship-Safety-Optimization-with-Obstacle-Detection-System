import torch
from pathlib import Path
from PIL import Image
from torchvision.transforms import functional as F
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
import pandas as pd

# Load YOLOv5 model
weights_path = 'yolov5s.pt'
model = attempt_load(weights_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Load classes
classes_df = pd.read_csv('classes.csv', header=None)
classes = classes_df.iloc[:, 0].tolist()

# Load dataset and annotations
folder_path = r'C:\Users\syari\Downloads\AutoLabelling.v1i.tensorflow\validYOLO'
annotations_path = 'annotations.csv'
annotations_df = pd.read_csv(annotations_path, header=None)

# Detection loop
for index, row in annotations_df.iterrows():
    image_path = Path(folder_path) / row[0]
    bboxes = [list(map(int, box.split(','))) for box in row[1:]]
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Preprocess image
    img_tensor = F.to_tensor(img).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        pred = model(img_tensor)[0]

    # Post-process predictions
    pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.5)[0]

    # Scale bounding boxes to the original image size
    img_shape = img.shape
    pred[:, :4] = scale_coords(img_tensor.shape[2:], pred[:, :4], img_shape).round()

    # Visualize or save the results
    for det in pred:
        label = int(det[-1])
        box = det[:4].cpu().numpy().astype(int)
        plot_one_box(box, img, label=f'{classes[label]}', color=(0, 255, 0), line_thickness=2)

    img.show()
