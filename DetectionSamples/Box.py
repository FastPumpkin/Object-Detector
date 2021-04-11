import numpy as np
import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
class BoxCreator(object):
    def __init__(self, image, window_name,color=(0,0,255)):
        self.image = image
        self.orig = image.copy()
        self.start = None
        self.end = None        
        self.track = False
        self.color = color
        self.window_name = window_name
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name,self.mouseCallBack)

    def mouseCallBack(self, event, x, y, flags, params):
        if event==cv2.EVENT_LBUTTONDOWN:
            self.start = (x,y)
            self.track = True
        elif self.track and (event==cv2.EVENT_MOUSEMOVE or event==cv2.EVENT_LBUTTONUP):
            self.end = (x,y)
            if not self.start==self.end:
                self.image = self.orig.copy()
                cv2.rectangle(self.image, self.start, self.end, self.color, 2)
                if event==cv2.EVENT_LBUTTONUP:
                    self.track=False
            else:
                self.image = self.orig.copy()
                self.start = None
                self.track = False
            cv2.imshow(self.window_name,self.image)

    
    @property
    def roiPts(self):
       if self.start and self.end:
           pts = np.array([self.start,self.end])
           s = np.sum(pts,axis=1)
           (x,y) = pts[np.argmin(s)]
           (xb,yb) = pts[np.argmax(s)]
           return [(x,y),(xb,yb)]
       else:
           return []
