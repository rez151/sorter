import cv2


class VideoHandler:
    # video_source kann auch variieren!!! bei mir zwischen 1 und 5
    # sonst ist hier nichts zu machen
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return ret, None
        else:
            return None, None

    def save_frame(self, name):
        ret, frame = self.get_frame()
        cv2.imwrite(name, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # cv2.imwrite(name, frame)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
