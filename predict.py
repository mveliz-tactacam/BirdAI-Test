import darknet
from PIL import Image #// pip install pillow

net = darknet.load_net(b"cfg/yolov4.cfg", b"feathersnap.weights", 0)
meta = darknet.load_meta(b"cfg/coco.data")
class_names = [meta.names[i].decode("ascii") for i in range(meta.classes)]

def detect_objects(image_path):
    # Load image
    image = Image.open(image_path)
    image_rgb = image.convert("RGB")
    width, height = image_rgb.size

    # Convert image to Darknet format
    darknet_image = darknet.make_image(width, height, 3)
    image_data = image_rgb.tobytes()
    darknet.copy_image_from_bytes(darknet_image, image_data)

    # Perform object detection
    detections = darknet.detect_image(net, class_names, darknet_image)

    # Release resources
    darknet.free_image(darknet_image)

    bird_detection = get_bird_detection(detections)

    if bird_detection[0] != None:
      crop_to_bbox(bird_detection[0], image_rgb)
      print("Bird detected with confidence: ", bird_detection[1])
    
    print("Bird detected: ", bird_detection[0] != None)
    return [bird_detection[0] != None, bird_detection[1]]


def get_bird_detection(detections):
    max_confidence = 0.0
    max_confidence_detection = None
    for detection in detections:
        label, confidence, _ = detection
        if label == 'bird':
            confidence_value = float(confidence)
            if confidence_value > max_confidence:
                max_confidence = confidence_value
                max_confidence_detection = detection
    if max_confidence_detection:
        return [detection, max_confidence]
    else:
        return [None, 0.0]

def crop_to_bbox(detection, image_rgb):
    label, confidence, bbox = detection
    print(confidence)
    x, y, w, h = bbox
    x1 = int(x - w / 2)
    y1 = int(y - h / 2)
    x2 = int(x + w / 2)
    y2 = int(y + h / 2)
    cropped_image = image_rgb.crop((x1, y1, x2, y2))
    cropped_image.save('results/prediction.png') # Instead of saving, send back to backend in some form

image_path = "data/notbird3.png" # Replace with image from backend
results = detect_objects(image_path) # Run this to detect image