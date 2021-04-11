# Object-Detector
Custom Object Detector using dlib

To test the model trained copy and paste the following command:

      $ python test.py

To get a new model from scratch first run boxes_creator.py for train set and test set as follow:

      $ python boxes_creator.py -d [Train set path] -a [Path to save train boxes annotations (.npy)] -i [Path to save  train images annotations (.npy)]
      
      $ python boxes_creator.py -d [Test set path] -a [Path to save test boxes annotations (.npy)] -i [Path to save test images annotations (.npy)]

Use your mouse to create a rectangle that contains the objective and press esc to go the next image.

Then, run train.py programm using the next command:

      $ python train.py -a [Path to saved train boxes annotations] -i [Path to saved train images annotations] -d [Path to save model (.svm)] -ta [Path to saved test boxes annotations] -tim [Path to saved test images annotations]
      
To test, run:

      $ python test.py -d [Path to trained model] -a [Name of the object to detect]
