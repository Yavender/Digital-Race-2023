import numpy as np
import torch
import cv2
import time
import segmentation_models_pytorch as smp

COLORMAP = [
    [0, 0, 0],
    [255, 0, 0],
    [255, 255, 255],
]
model_path = "modelPSPNet_ep_71.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.create_model("pspnet", "timm-mobilenetv3_small_minimal_100", "imagenet", 3, 3).to(device)
model.load_state_dict(torch.load(model_path, device))
model.eval()

# Define mean and std values for normalization (replace with your actual values)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def apply_colormap(segmentation_result):
    color_mask_predict = np.zeros((*segmentation_result.shape, 3), dtype=np.uint8)
    for i, color in enumerate(COLORMAP):
        color_mask_predict[segmentation_result == i] = np.array(color)
    return color_mask_predict

def segment(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Preprocess the input frame (resize and normalize)
    frame_rgb = cv2.resize(frame_rgb, (128, 128), interpolation=cv2.INTER_LINEAR)
    frame_rgb = frame_rgb.astype('float32') / 255.0
    frame_rgb = (frame_rgb - mean) / std  # Normalize
    frame_rgb = frame_rgb.transpose(2, 0, 1)
    x = torch.from_numpy(frame_rgb).unsqueeze(0).to(device).float()

    with torch.no_grad():
        start_time = time.time()  # Thời điểm bắt đầu xử lý video
        y_predict = model(x).argmax(dim=1).squeeze().cpu().numpy()
        elapsed_time = time.time() - start_time

        color_mask_predict = apply_colormap(y_predict)
    print(elapsed_time)
    color_mask_predict = cv2.resize(color_mask_predict, (640, 480), interpolation=cv2.INTER_LINEAR)
    color_mask_predict = cv2.cvtColor(color_mask_predict, cv2.COLOR_BGR2RGB)
    cv2.imshow("Semantic Segmentation", color_mask_predict)
    return color_mask_predict

cap = cv2.VideoCapture("vide.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    segment(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()