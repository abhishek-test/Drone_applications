import cv2

# Replace 'your_gstreamer_pipeline' with the actual GStreamer pipeline
gstreamer_pipeline = "udpsrc port=5000 caps=application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96 ! rtph264depay ! queue ! h264parse ! queue ! avdec_h264 ! autovideosink"

# Open the video stream using the GStreamer pipeline
cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error reading frame")
        break

    cv2.imshow('GStreamer Stream', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

