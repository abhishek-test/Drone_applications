import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import tkinter as tk
import threading

Gst.init(None)

class VideoPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("RTSP Video Player")
        self.pipeline = None
        
        self.setup_pipeline()
        
        self.play_button = tk.Button(self.root, text="Play", command=self.start_pipeline)
        self.play_button.pack(pady=10)
        
        self.stop_button = tk.Button(self.root, text="Stop", command=self.stop_pipeline)
        self.stop_button.pack(pady=5)
        
    def setup_pipeline(self):
        self.pipeline = Gst.parse_launch("udpsrc port=5000 caps=application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96 ! rtph264depay ! queue ! h264parse ! queue ! avdec_h264 ! autovideosink")
    
    def start_pipeline(self):
        self.pipeline.set_state(Gst.State.PLAYING)
        
    def stop_pipeline(self):
        self.pipeline.set_state(Gst.State.NULL)
        
def gui_thread():
    root = tk.Tk()
    player = VideoPlayer(root)
    root.mainloop()

if __name__ == "__main__":
    threading.Thread(target=gui_thread).start()

