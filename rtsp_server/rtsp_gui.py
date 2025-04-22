import gi
gi.require_version('Gst', '1.0')
gi.require_version('Gtk', '3.0')
from gi.repository import Gst, Gtk, Gdk, GObject, GstVideo
import threading

Gst.init(None)
GObject.threads_init()

class VideoPlayer:
    def __init__(self):
        self.window = Gtk.Window()
        self.window.set_default_size(1280, 720)  # Adjust the size for 720p
        self.window.connect("destroy", self.quit)

        self.play_button = Gtk.Button(label="Play")
        self.play_button.connect("clicked", self.start_pipeline)

        self.stop_button = Gtk.Button(label="Stop")
        self.stop_button.connect("clicked", self.stop_pipeline)

        self.drawing_area = Gtk.DrawingArea()
        self.drawing_area.set_size_request(1280, 720)  # Adjust the size for 720p
        self.drawing_area.connect("draw", self.on_draw)

        self.pixbuf = None

        self.grid = Gtk.Grid()
        self.grid.attach(self.play_button, 0, 0, 1, 1)
        self.grid.attach(self.stop_button, 1, 0, 1, 1)
        self.grid.attach(self.drawing_area, 0, 1, 2, 1)

        self.window.add(self.grid)

        self.setup_pipeline()

    def setup_pipeline(self):
        pipeline_string = (
            "udpsrc port=5000 ! "
            "application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96 ! "
            "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink name=sink"
        )
        self.pipeline = Gst.parse_launch(pipeline_string)

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::eos", self.on_eos)

        self.appsink = self.pipeline.get_by_name("sink")
        self.appsink.set_property("emit-signals", True)
        self.appsink.connect("new-sample", self.on_new_sample)

    def start_pipeline(self, button):
        self.pipeline.set_state(Gst.State.PLAYING)

    def stop_pipeline(self, button):
        self.pipeline.set_state(Gst.State.NULL)

    def on_eos(self, bus, message):
        self.pipeline.set_state(Gst.State.NULL)

    def on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        buffer = sample.get_buffer()

        caps = sample.get_caps()
        structure = caps.get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")

        video_frame = GstVideo.VideoFrame.new(buffer, caps, 0, 0, width, height)

        pixbuf = Gdk.pixbuf_new(
            Gdk.Colorspace.RGB,
            True,  # Use alpha channel for color input
            8,
            width,
            height,
        )

        video_frame.extract_into(pixbuf.get_pixels(), 0, pixbuf.get_rowstride() * height)

        self.pixbuf = pixbuf

        self.drawing_area.queue_draw()

        return Gst.FlowReturn.OK

    def on_draw(self, widget, cr):
        if self.pixbuf is not None:
            Gdk.cairo_set_source_pixbuf(cr, self.pixbuf, 0, 0)
            cr.paint()
        else:
	        return False

    def run(self):
        self.window.show_all()
        Gtk.main()

    def quit(self, *args):
        self.pipeline.set_state(Gst.State.NULL)
        Gtk.main_quit()

def run_gui():
    player = VideoPlayer()
    player.run()

if __name__ == "__main__":
    threading.Thread(target=run_gui).start()

