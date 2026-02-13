import cv2
import numpy as np
from PIL import Image

def enhance_cctv_image(image_path, output_path=None):
    """
    Apply specialized enhancements for CCTV images
    """
    # Read image
    img = cv2.imread(image_path)
    
    if img is None:
        return None
    
    # Convert to grayscale for some operations
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Convert back to color
    enhanced_color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
    # Combine original and enhanced
    alpha = 0.7
    beta = 0.3
    final = cv2.addWeighted(denoised, alpha, enhanced_color, beta, 0)
    
    if output_path:
        cv2.imwrite(output_path, final)
    
    return final

def detect_objects(image_path):
    """
    Basic object detection for CCTV images
    """
    # This is a placeholder - in production, you might want to use
    # YOLO, Detectron2, or similar for better object detection
    import torch
    from torchvision import models, transforms
    
    # Load pretrained model
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    
    # Preprocess image
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    
    # Predict
    with torch.no_grad():
        predictions = model([image_tensor])
    
    # COCO class labels
    coco_labels = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
        'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
        'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
        'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    detected_objects = []
    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']
    
    for i in range(min(5, len(boxes))):  # Get top 5 detections
        if scores[i] > 0.5:  # Confidence threshold
            label = coco_labels[labels[i] - 1]
            detected_objects.append({
                'object': label,
                'confidence': float(scores[i])
            })
    
    return detected_objects

def get_image_metadata(image_path):
    """
    Extract metadata from image
    """
    from PIL import Image
    import exifread
    
    metadata = {}
    
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f)
        for tag in tags.keys():
            if tag not in ('JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote'):
                metadata[tag] = str(tags[tag])
    
    # Get basic info
    img = Image.open(image_path)
    metadata['size'] = img.size
    metadata['mode'] = img.mode
    metadata['format'] = img.format
    
    return metadata