import cv2
import torch
import numpy as np
from torchvision import transforms

# Parameters
MAX_CORNERS = 200
RESET_THRESHOLD   = 10  # If fewer than this number of points remain, reset
FRAME_HISTORY     = 10  # Number of frames to keep
AUTO_RESET_FRAMES = 100 # Reset tracking every 100 frames

# Load MiDaS Model (Small Version for Faster Inference)
def load_midas():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device)
    model.eval()
    return model, device

# Transform Input for MiDaS
def transform_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((384, 384)),  # Keep MiDaS input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Ensure 3-channel input
    ])
    return transform(image).unsqueeze(0)

# Compute Depth using MiDaS
def compute_depth(model, device, frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform_image(image).to(device)

    with torch.no_grad():
        depth_map = model(input_tensor)

    depth_map = depth_map.squeeze().cpu().numpy()
    
    # Normalize depth to maintain relative values
    depth_min = np.min(depth_map)
    depth_max = np.max(depth_map)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min + 1e-6) * 255
    depth_map = cv2.resize(depth_map.astype(np.uint8), (frame.shape[1], frame.shape[0]))
    depth_map_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
    depth_map = 255 - depth_map

    return depth_map, depth_map_color

# Function to detect new feature points
def detect_features(frame_gray):
    return cv2.goodFeaturesToTrack(frame_gray, maxCorners=MAX_CORNERS, qualityLevel=0.01, minDistance=10)

# Compute Optical Flow using Farneback
def compute_optical_flow(prev_gray, gray):
    #return cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
    return cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 5, 15, 3, 7, 1.5, 0)
    
# Compute Focus of Expansion (FoE)
def compute_focus_of_expansion(flow, depth):
    h, w = flow.shape[:2]
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))

    x_coords, y_coords = x_coords.flatten(), y_coords.flatten()
    flow_x, flow_y = flow[..., 0].flatten(), flow[..., 1].flatten()
    depth = depth.flatten()

    # Normalize depth to remove extreme values
    depth = np.clip(depth, 1e-6, np.percentile(depth, 99))  # Remove outliers
    weights = 1.0 / depth  # Compute inverse depth weighting

    # Apply weights to motion vectors
    flow_x *= weights
    flow_y *= weights

    # Least Squares to find FoE
    A = np.column_stack((flow_x, flow_y))
    b = -(x_coords * flow_x + y_coords * flow_y)

    try:
        FoE_x, FoE_y = np.linalg.lstsq(A, b, rcond=None)[0]
        FoE_x, FoE_y = int(FoE_x), int(FoE_y)
        
        # If FoE is outside the image, fallback to center
        if FoE_x < 0 or FoE_y < 0 or FoE_x > w or FoE_y > h:
            raise ValueError
    except:
        FoE_x, FoE_y = w // 2, h // 2  # Default to image center

    return FoE_x, FoE_y

# Detect Collision Based on FoE & Depth
def detect_collision(FoE_x, FoE_y, depth_map, threshold=50):
    h, w = depth_map.shape
    center_x, center_y = w // 2, h // 2

    window = depth_map[63:213, 250:400]

    # If FoE is near the center and depth is below threshold, obstacle detected
    #if abs(FoE_x - center_x) < 50 and abs(FoE_y - center_y) < 50:
    if np.min(window) < threshold:
        return True  
    return False

# Video Processing Loop
cap = cv2.VideoCapture("video_1.mp4")
ret, prev_frame = cap.read()
prev_frame = cv2.resize(prev_frame, (640, 360), interpolation=cv2.INTER_AREA)
prev_gray  = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Load MiDaS
midas_model, device = load_midas()

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
    if not ret:
        break

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow  = compute_optical_flow(prev_gray, gray)
    depth, depth_map_color = compute_depth(midas_model, device, frame)
    FoE_x, FoE_y = compute_focus_of_expansion(flow, depth)

    # Detect Collision
    collision = detect_collision(FoE_x, FoE_y, depth)
    collision_text = "Collision Detected!" if collision else "No Collision"

    # Draw Optical Flow
    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Draw Focus of Expansion (FoE)
    #cv2.circle(flow_vis, (FoE_x, FoE_y), 5, (0, 0, 255), -1)
    #cv2.circle(frame, (FoE_x, FoE_y), 5, (0, 0, 255), -1)

    #cv2.circle(depth_map_color, (FoE_x, FoE_y), 5, (0, 0, 255), -1)
    cv2.rectangle(depth_map_color, (250, 63), (400, 213), (0,0,0), 2, 8)
    cv2.putText(depth_map_color, collision_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if collision else (0, 255, 0), 2)

    # Show Results
    cv2.imshow("Original ", frame)
    cv2.imshow("Depth", depth_map_color)
    #cv2.imshow("Flow", flow_vis)
    
    prev_gray = gray

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
