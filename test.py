from detector import ObjectDetector
import numpy as np
import cv2
import os
import glob
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-d","--detector", help="Path to trained model", default="svm_model.svm")
ap.add_argument("-a","--annotate", help="Object Name", default="Object")

args = vars(ap.parse_args())
detector = ObjectDetector(loadPath=args["detector"])


path = os.getcwd()
pathDir = path + "/TestImages/"

testImages = glob.glob(pathDir + "*")


cap = cv2.VideoCapture(0)
bandera = 0
while True:
	if bandera == 0:
		for fnames in testImages:
			image = cv2.imread(fnames)
			detector.detect(image,annotate=args["annotate"])
			cv2.waitKey(1000)
		bandera = 1		
	flag, image = cap.read()
	detector.detect(image,annotate=args["annotate"])

