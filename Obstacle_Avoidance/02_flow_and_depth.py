import cv2
import numpy as np
import torch
from collections import deque
from torchvision import transforms

# Load video
cap = cv2.VideoCapture("video_2.mp4")

# Parameters
MAX_CORNERS = 200
RESET_THRESHOLD = 50  # Reset if fewer than this many points remain
FRAME_HISTORY = 10    # Keep optical flow history for the last 15 frames
AUTO_RESET_FRAMES = 1000  # Reset tracking every 1000 frames    

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

# Read first frame
ret, prev_frame = cap.read()
prev_frame = cv2.resize(prev_frame, (640, 360), interpolation=cv2.INTER_AREA)
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Function to detect new feature points
def detect_features(frame_gray):
    features = cv2.goodFeaturesToTrack(frame_gray, maxCorners=MAX_CORNERS, qualityLevel=0.001, minDistance=10)
    return features if features is not None else np.array([])

# Initialize points and history
prev_pts = detect_features(prev_gray)
flow_history = deque(maxlen=FRAME_HISTORY)
frame_count = 0  # Frame counter

# Compute Focus of Expansion (FoE) using Least Squares from history
def compute_focus_of_expansion(flow_history):
    if not flow_history:
        return None
    
    # Concatenate all flow vectors from the history (each vector: ((x_new, y_new), (x_old, y_old)))
    all_vectors = np.concatenate(flow_history, axis=0)
    if len(all_vectors) < 5:  # Not enough data for estimation
        return None

    # Separate start points and end points
    start_pts = np.array([vec[1] for vec in all_vectors])
    end_pts = np.array([vec[0] for vec in all_vectors])

    # Compute flow directions
    flow = end_pts - start_pts
    A = np.column_stack((flow[:, 0], flow[:, 1]))
    b = -(start_pts[:, 0] * flow[:, 0] + start_pts[:, 1] * flow[:, 1])
    try:
        FoE_x, FoE_y = np.linalg.lstsq(A, b, rcond=None)[0]
        #FoE_x, FoE_y = np.linalg.lstsq(b, A, rcond=None)[0]
        return abs(int(FoE_x)), abs(int(FoE_y))
    except:
        return None
    
# Load MiDaS
midas_model, device = load_midas()

while cap.isOpened():
    ret, curr_frame = cap.read()
    if not ret:
        break

    curr_frame = cv2.resize(curr_frame, (640, 360), interpolation=cv2.INTER_AREA)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    frame_count += 1  # Increment frame counter

    # calculate depth from Midas
    depth, depth_map_color = compute_depth(midas_model, device, curr_frame)

    # Compute optical flow using Lucas-Kanade
    if prev_pts.size != 0:
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
        if curr_pts is None:
            prev_pts = detect_features(curr_gray)
            continue

        # Select good points
        good_new = curr_pts[status.flatten() == 1]
        good_old = prev_pts[status.flatten() == 1]

        if len(good_new) == 0:
            prev_pts = detect_features(curr_gray)
            continue

        # Build motion vectors from current frame:
        # For each pair, flatten the points and then create a tuple:
        frame_vectors = [((p_new.ravel()[0], p_new.ravel()[1]),
                           (p_old.ravel()[0], p_old.ravel()[1]))
                         for p_new, p_old in zip(good_new, good_old)]
        flow_history.append(frame_vectors)

        # Compute FoE from the history of optical flow vectors
        FoE = compute_focus_of_expansion(flow_history)

        # Create an empty mask to draw vectors
        mask = np.zeros_like(curr_frame)

        # Draw flow vectors from the history
        for i, vectors in enumerate(flow_history):
            # Fade color for older frames
            color = (0, 255 - (i * int(255 / FRAME_HISTORY)), 0)
            for (x_new, y_new), (x_old, y_old) in vectors:
                cv2.line(mask, (int(x_new), int(y_new)), (int(x_old), int(y_old)), color, 2)
                cv2.circle(mask, (int(x_new), int(y_new)), 1, (0, 0, 255), -1)

        # Draw FoE if computed
        if FoE is not None:
            cv2.circle(curr_frame, FoE, 10, (0, 0, 0), -1)
            cv2.putText(curr_frame, "FoE: (" + str(FoE[0]) + "," + str(FoE[1]) + ")", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        prev_gray = curr_gray.copy()
        prev_pts = good_new.reshape(-1, 1, 2)

        # Overlay flow vectors on the original frame
        output = cv2.add(curr_frame, mask)
        #cv2.putText(output, str(len(good_new)), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.imshow("Sparse Optical Flow", output)  
        cv2.imshow("Depth Map", depth_map_color)

        # Reset points if too few remain or after AUTO_RESET_FRAMES
        if len(good_new) < RESET_THRESHOLD or frame_count >= AUTO_RESET_FRAMES:
            prev_pts = detect_features(curr_gray)
            flow_history.clear()
            frame_count = 0
            #print("🔄 Auto-reset: Detecting new points")

        # Manual reset (Press 'r')
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            prev_pts = detect_features(curr_gray)
            flow_history.clear()
            frame_count = 0
            #print("🆕 Manual reset: New feature points")
        if key == ord('p'):
            cv2.waitKey(0)
        if key == ord('q'):
            break

    else:
        # If no points, detect features
        prev_pts = detect_features(prev_gray)

cap.release()
cv2.destroyAllWindows()
