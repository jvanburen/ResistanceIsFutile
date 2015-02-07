from __future__ import with_statement, division
import cv2
import sys
import pprint
import time
import numpy as np
import threading
from scipy.ndimage import label
import resist2 as colors

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


cascPath = './resistor/cascade.xml'
resCascade = cv2.CascadeClassifier(cascPath)

width = 600
height = 480

video_capture = cv2.VideoCapture(0)
video_capture.set(3, width)
video_capture.set(4, height)
 
webcam = MyWebcam(video_capture)
webcam.start()

screen_center_x = width//2
screen_center_y = height//2



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

def segment_on_dt(a, img):
    border = cv2.dilate(img, None, iterations=5)
    border = border - cv2.erode(border, None)

    dt = cv2.distanceTransform(img, 2, 3)
    dt = ((dt - dt.min()) // (dt.max() - dt.min()) * 255).astype(np.uint8)
    _, dt = cv2.threshold(dt, 180, 255, cv2.THRESH_BINARY)
    lbl, ncc = label(dt)
    lbl = lbl * (255/ncc)
    # Completing the markers now. 
    lbl[border == 255] = 255

    #import pdb; pdb.set_trace()

    lbl = lbl.astype(np.int32)
    
    cv2.watershed(a, lbl)

    lbl[lbl == -1] = 0
    lbl = lbl.astype(np.uint8)
    return 255 - lbl
    
prev = []
while True:
    ret, frame = webcam.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # features = []
    features = resCascade.detectMultiScale(
             gray,
             scaleFactor=1.1,
             minNeighbors=50,
             minSize=(8, 3),
             maxSize=(width//4, height//4),
             flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    
    if len(features) > 0:
        
       	x, y, w, h = max(features, key=lambda f: f[2]*f[3])
        
        # img_gray = gray[x:x+width, y:y+height] 
        # _, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)
        # img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, np.ones((3, 3), dtype=int))
        # result = segment_on_dt(frame[x:x+width, y:y+height], img_bin)
        
        # gray[x:x+width, y:y+height] = result
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
       	# print x,y,w,h
    else:
        x, y, w, h = screen_center_x - 75, screen_center_y-28, 150, 56
    
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
    
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    prev = features
    

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
import time
time.sleep(0.5)
exit(0)
