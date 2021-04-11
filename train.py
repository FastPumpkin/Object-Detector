from detector import ObjectDetector
import numpy as np
import argparse
parse = argparse.ArgumentParser(description="Object Detector Training Program")
parse.add_argument("-a","--annotations",help=" Path to saved boxes annotations", default="train_annot.npy")
parse.add_argument("-i","--images",help="Path to saved images", default="train_images.npy")
parse.add_argument("-d","--detector",help="Path to save model", default="svm_model.svm")
parse.add_argument("-ta","--test_annotate",help="Path to saved test boxes annotations", default="test_annot.npy")
parse.add_argument("-tim","--test_images",help="Path to test images", default="test_images.npy")
args = vars(parse.parse_args())

annots = np.load(args["annotations"])
imagePaths = np.load(args["images"])
trainAnnot = np.load(args["test_annotate"])
trainImages = np.load(args["test_images"])

detector = ObjectDetector()
detector.fit(imagePaths,annots,trainAnnot,trainImages,visualize=True,savePath=args["detector"])
