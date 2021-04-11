import numpy as np
from DetectionSamples.Box import BoxCreator
import cv2
import argparse
from imutils.paths import list_images

#Argumentos necesarios para la ejecucion
ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help="Path to train or test images")
ap.add_argument("-a","--annotations",required=True,help="Path to save boxes annotations")
ap.add_argument("-i","--images",required=True,help="Pat to save image annotations")
args = vars(ap.parse_args())

#Vectores vacios para imagenes y anotaciones
annotations = []
imPaths = []
#Loop para cada imagen presente en el dataset
for imagePath in list_images(args["dataset"]):
    #Se carga la imagen y se guardan las coordenadas 
    #del rectangulo que encierra al objeto
    image = cv2.imread(imagePath)
    bs = BoxCreator(image,"Image")
    cv2.imshow("Image",image)
    cv2.waitKey(0)
    #order the points suitable for the Object detector
    pt1,pt2 = bs.roiPts
    (x,y,xb,yb) = [pt1[0],pt1[1],pt2[0],pt2[1]]

    annotations.append([int(x),int(y),int(xb),int(yb)])
    print(x,y,xb,yb)
    imPaths.append(imagePath)
    print(imagePath)

#Se guardan las coordenadas y las imagenes en archivos
annotations = np.array(annotations)
imPaths = np.array(imPaths,dtype="unicode")
np.save(args["annotations"],annotations)
np.save(args["images"],imPaths)
