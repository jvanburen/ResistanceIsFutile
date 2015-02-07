from __future__ import with_statement
import cv2
import sys
import pprint
import time
import numpy as np
import threading

class MyWebcam(threading.Thread):
    def __init__(self, vid):
        threading.Thread.__init__(self)
        self.daemon = True
        self.vid = vid
        self.lock = threading.Lock()
        self.image = None

    def run(self):
        while 1:
            frame = self.vid.read()
            with self.lock:
                self.image = frame

    def read(self):
        with self.lock:
            image = self.image
        return image



cascPath = './classifier/cascade.xml'
resCascade = cv2.CascadeClassifier(cascPath)


width = 640
height = 480

video_capture = cv2.VideoCapture(0)
video_capture.set(3, width)
video_capture.set(4, height)
 
webcam = MyWebcam(video_capture)
webcam.start()

screen_center_x = width/2
screen_center_y = height/2

def compute_colour_distance(image, x, y, w, h, color):
    part = image[x:x+w, y:y+h]
    hsv = part.astype('float32') * 1./255
    hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV);
    
    

def compute_blurriness_coeff(frame, cvtcolor=True):
    "WARNING: slow-ish!"
    #a good metric is probably blurriness < 70 means it's sharp

    LAPLACIAN_KERNEL = np.zeros((3,3), 'float32')
    LAPLACIAN_KERNEL[0,1] = 1
    LAPLACIAN_KERNEL[1,0] = 1
    LAPLACIAN_KERNEL[1,2] = 1
    LAPLACIAN_KERNEL[2,1] = 1
    LAPLACIAN_KERNEL[1,1] = -4

    if cvtcolor:
        gray = frame.astype('float32') * 1./255
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()
    gray = cv2.filter2D(gray, ddepth=-1,kernel=LAPLACIAN_KERNEL)
    gray = gray.reshape(-1)

    return 100./max(iter(gray))




def test_camera(vid):
    print ("starting camera")
    while True:
        ret, frame = vid.read()
        blurriness = compute_blurriness_coeff(frame)
        info = "blurriness: %0.2f" % blurriness
        cv2.putText(frame, info, (0,frame.shape[0]),
            cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0))
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
            
def read_grayscale(vid):
    return cv2.cvtColor(
                        vid.read()[-1].astype('float32') * 1./255,
                        cv2.COLOR_BGR2GRAY)

def get_clear_image(vid, threshold=(0,100), delay=0.005, verbosity=0):
    frame = read_grayscale(vid)
    blurriness = compute_blurriness_coeff(frame, False)
    while threshold[0] >= blurriness or blurriness >= threshold[1]:
        if verbosity >= 1: print "took picture with blurriness %0.3f" % blurriness
        if verbosity >= 2:
            info = "blurriness: %0.2f" % blurriness
            cv2.putText(frame, info, (0,frame.shape[0]),
                        cv2.FONT_HERSHEY_DUPLEX, 1, 0)
            while 1:
                cv2.imshow("get_clear_image", frame)
                if cv2.waitKey(1) & 0xFF:
                    break
        time.sleep(0.005)
        frame = read_grayscale(vid)
        blurriness = compute_blurriness_coeff(frame, False)

    if verbosity >= 2:
        cv2.putText(frame, "CLEAR IMAGE: %0.2f"% blurriness,
                        (0,frame.shape[0]),
                        cv2.FONT_HERSHEY_DUPLEX, 1, 0)
        while 1:
            cv2.imshow("get_clear_image", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    return frame



test_camera(video_capture)

while True:
    ret, frame = webcam.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # features = []
    features = resCascade.detectMultiScale(
             gray,
             scaleFactor=1.1,
             minNeighbors=7,
             minSize=(5, 5),
             flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    
    if len(features) > 0:
       	x, y, w, h = max(features, key=lambda f: f[2]*f[3])
       	# print x,y,w,h
    else:
        x, y, w, h = screen_center_x - 75, screen_center_y-28, 150, 56
    
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
exit(0)