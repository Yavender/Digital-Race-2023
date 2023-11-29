import albumentations as A
from albumentations.pytorch import ToTensorV2 # np.array -> torch.tensor

CLASSES = [
    "background",
    "road",
    "line"
]

COLORMAP = [
    [0, 0, 0],
    [255, 0, 0],
    [255, 255, 255],
]

train_size = 256

train_transform = A.Compose([
    A.Resize(width=train_size, height=train_size),
    A.HorizontalFlip(),
    A.RandomBrightnessContrast(),
    A.Blur(),
    A.Sharpen(),
    A.RGBShift(),
    A.Cutout(num_holes=5, max_h_size=25, max_w_size=25, fill_value=0),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
    ToTensorV2(),
])

test_transform = A.Compose([
    A.Resize(width=train_size, height=train_size),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
    ToTensorV2(), # numpy.array -> torch.tensor (B, 3, H, W)
])


model_path = "Road_Line_Segmentation/modelPSPNet_ep_59.pth"
video_path = "video/vide.mp4"