import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import requests # To download an image

# Check if CUDA is available and set the device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

# 1. Load a pre-trained Faster R-CNN model
#    (pretrained=True is deprecated, use weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
try:
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
except ImportError:
    # Fallback for older torchvision versions
    model = fasterrcnn_resnet50_fpn(pretrained=True)

model.eval()  # Set the model to evaluation mode
model.to(device) # Move model to the selected device

# 2. Define COCO class names (Faster R-CNN is often trained on COCO)
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# 3. Prepare the image
#    You can replace this URL with a path to your local image
# img_url = 'http://images.cocodataset.org/val2017/000000039769.jpg' # Example image: cats and a remote
try:
    # img = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    img = Image.open(r"../data/sandwich.png").convert("RGB")
except requests.exceptions.RequestException as e:
    print(f"Error downloading image: {e}")
    print("Please use a local image path instead.")
    # Example: img = Image.open("path/to/your/image.jpg").convert("RGB")
    exit()


# Define the image transformation
transform = T.Compose([T.ToTensor()])
img_tensor = transform(img).to(device)

# 4. Perform inference
with torch.no_grad(): # No need to track gradients for inference
    prediction = model([img_tensor])

# 5. Process and display results
#    prediction is a list of dicts. For a single image, it will be prediction[0].
#    prediction[0]['boxes'] are the bounding boxes [xmin, ymin, xmax, ymax]
#    prediction[0]['labels'] are the predicted class IDs
#    prediction[0]['scores'] are the confidence scores

boxes = prediction[0]['boxes'].cpu().numpy()
labels = prediction[0]['labels'].cpu().numpy()
scores = prediction[0]['scores'].cpu().numpy()

# Filter out predictions with low confidence
confidence_threshold = 0.5
filtered_indices = scores > confidence_threshold
filtered_boxes = boxes[filtered_indices]
filtered_labels = labels[filtered_indices]
filtered_scores = scores[filtered_indices]

# Plot the image and bounding boxes
fig, ax = plt.subplots(1, figsize=(12, 9))
ax.imshow(img)

for box, label_id, score in zip(filtered_boxes, filtered_labels, filtered_scores):
    xmin, ymin, xmax, ymax = box
    label_name = COCO_INSTANCE_CATEGORY_NAMES[label_id]
    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.text(xmin, ymin - 5, f"{label_name}: {score:.2f}",
             bbox=dict(facecolor='yellow', alpha=0.5), fontsize=10, color='black')

plt.axis('off')
plt.show()

print("Detected objects and their scores:")
for i in range(len(filtered_labels)):
    print(f"- {COCO_INSTANCE_CATEGORY_NAMES[filtered_labels[i]]}: {filtered_scores[i]:.2f}")