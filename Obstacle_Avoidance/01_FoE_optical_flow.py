import cv2
import numpy as np
from collections import deque
from sklearn.linear_model import RANSACRegressor
from filterpy.kalman import KalmanFilter  # Import Kalman Filter

# Load video
cap = cv2.VideoCapture("video_2.mp4")

# Parameters
MAX_CORNERS = 200
RESET_THRESHOLD = 100  # Reset if fewer than this many points remain
FRAME_HISTORY = 10  # Keep optical flow history for the last 15 frames
AUTO_RESET_FRAMES = 1000  # Reset tracking every 1000 frames

# Initialize Kalman Filter for FoE
kf_foe = KalmanFilter(dim_x=4, dim_z=2)  # State: [FoE_x, FoE_y, vFoE_x, vFoE_y], Measurement: [FoE_x, FoE_y]
kf_foe.x = np.array([[0], [0], [0], [0]])  # Initial state (FoE position and velocity)
kf_foe.P *= 1000.  # Initial uncertainty
kf_foe.F = np.array([[1, 0, 1, 0],  # State transition matrix
                      [0, 1, 0, 1],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
kf_foe.H = np.array([[1, 0, 0, 0],  # Measurement function
                      [0, 1, 0, 0]])
kf_foe.R = np.array([[1000, 0],  # Measurement noise
                      [0, 1000]])
kf_foe.Q = np.eye(4) * 0.1  # Process noise

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

'''
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
        #FoE_x, FoE_y = np.linalg.lstsq(A, b, rcond=None)[0]
        FoE_x, FoE_y = np.linalg.lstsq(b, A, rcond=None)[0]
        return int(FoE_x), int(FoE_y)
    except:
        return None
'''
    
import numpy as np

def compute_focus_of_expansion(frame_vectors):
    """
    Compute Focus of Expansion (FoE) using Least Squares.
    
    :param frame_vectors: List of ((x_new, y_new), (x_old, y_old)) tuples.
    :return: (FoE_x, FoE_y)
    """
    if len(frame_vectors) < 5:  # Need enough points for a robust estimate
        return None, None

    # Convert lists to NumPy arrays
    old_pts = np.array([pt[1] for pt in frame_vectors])  # (x_old, y_old)
    new_pts = np.array([pt[0] for pt in frame_vectors])  # (x_new, y_new)

    # Compute flow vectors
    flow_vectors = new_pts - old_pts
    flow_x, flow_y = flow_vectors[:, 0], flow_vectors[:, 1]
    x_old, y_old = old_pts[:, 0], old_pts[:, 1]

    if np.mean(flow_x) < 0:  
        flow_x *= -1  # Flip sign
    if np.mean(flow_y) < 0:
        flow_y *= -1

    avg_flow_x = np.mean(flow_x)
    avg_flow_y = np.mean(flow_y)

    # Subtract mean flow (to remove global motion like rotation)
    flow_x -= avg_flow_x
    flow_y -= avg_flow_y

    # Construct matrices for Least Squares
    A = np.column_stack((flow_x, flow_y))
    b = -(x_old * flow_x + y_old * flow_y)

    # Solve using Least Squares
    try:
        ransac = RANSACRegressor()
        ransac.fit(A, b)
        FoE_x, FoE_y = ransac.estimator_.coef_

        # Ensure FoE values are non-negative
        FoE_x = abs(int(FoE_x))
        FoE_y = abs(int(FoE_y))
    except:
        return None, None  # Return None if computation fails
    
    return int(FoE_x), int(FoE_y)



while cap.isOpened():
    ret, curr_frame = cap.read()
    if not ret:
        break

    curr_frame = cv2.resize(curr_frame, (640, 360), interpolation=cv2.INTER_AREA)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    frame_count += 1  # Increment frame counter

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
        #FoE = compute_focus_of_expansion(flow_history)
        #FoE = compute_focus_of_expansion(frame_vectors)

        h, w = curr_frame.shape[:2]
        center_x, center_y = w // 2, h // 2

        # Separate vectors into four quadrants
        quadrants = { "TL": [], "TR": [], "BL": [], "BR": [] }

        for (x_old, y_old), (x_new, y_new) in frame_vectors:
            if x_old < center_x and y_old < center_y:
                quadrants["TL"].append(((x_new, y_new), (x_old, y_old)))
            elif x_old >= center_x and y_old < center_y:
                quadrants["TR"].append(((x_new, y_new), (x_old, y_old)))
            elif x_old < center_x and y_old >= center_y:
                quadrants["BL"].append(((x_new, y_new), (x_old, y_old)))
            else:
                quadrants["BR"].append(((x_new, y_new), (x_old, y_old)))

        # Compute FoE for each quadrant (optional)
        foes = {}
        for key, vectors in quadrants.items():
            if len(vectors) > 5:
                foes[key] = compute_focus_of_expansion(vectors)

        # Find the quadrant with the most consistent FoE
        valid_foes = {k: v for k, v in foes.items() if v[0] is not None}
        if valid_foes:
            FoE_x, FoE_y = np.mean(list(valid_foes.values()), axis=0)
            FoE_x = int(FoE_x)
            FoE_y = int(FoE_y)

            # Update Kalman Filter with the new FoE measurements
            kf_foe.predict()  # Predict the next state
            kf_foe.update(np.array([[FoE_x], [FoE_y]]))  # Update with the FoE measurement

            # Get the predicted FoE position from the Kalman Filter
            predicted_foe_position = kf_foe.x[:2].reshape(-1).astype(int)

            # Draw the predicted FoE position
            cv2.circle(curr_frame, (predicted_foe_position[0], predicted_foe_position[1]), 5, (255, 0, 0), -1)


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
        #if FoE is not None:
        #cv2.circle(curr_frame, (FoE_x, FoE_y), 10, (0, 0, 0), -1)
        cv2.putText(curr_frame, "FoE: (" + str(FoE_x) + "," + str(FoE_y) + ")", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        prev_gray = curr_gray.copy()
        prev_pts = good_new.reshape(-1, 1, 2)

        # Overlay flow vectors on the original frame
        output = cv2.add(curr_frame, mask)
        #cv2.putText(output, str(len(good_new)), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.imshow("Sparse Optical Flow (Last 15 Frames)", output)

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
