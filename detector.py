import dlib
import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import time

#Claase Object Detector
class ObjectDetector(object):
	#########################################################
	#constructor de object Detector
    def __init__(self,options=None,loadPath=None):
        #Configuracion de las opciones de entrenamiento de un detector simple
        self.options = options
        if self.options is None:
        	#se crea el objeto
            self.options = dlib.simple_object_detector_training_options()
            #C es un parametro de regularizacion de SVM. Un valor grande ayuda al 
            #entrenamiento a tener un mejor ajuste a los datos. Sin embargo, esto puede
            #conducir a un sobre entrenamiento. El valor apropiado se encuentra de forma 
            #experimental
            self.options.C = 4.5
            # Si esta opcion es verdadera, el detector asumira que el objeto a detectar 
            #es simetrico y agrega imagenes volteadas de izquierda a derecha al set de
            #entrenamiento, esto produce que el traininset se duplique
            self.options.add_left_right_image_flips = True
            #Si es verdadero, se imprime informacion en la pantalla durante el entrenamiento
            self.options.be_verbose = False
            #Se define el numero de pixeles que contendra la ventana deslizante
            #8192 = 64x128p
            self.options.detection_window_size = 8192
            #Epsilon es un criterio de paro. Un valor pequeno provocara que el entrenamiento
            #sea mas preciso pero tomara mas tiempo
            self.options.epsilon = 0.0001
            #El tiempo maximo en segundos que debe tomar el entrenamiento
            #self.options.max_runtime_seconds = tiempo en segundos
            #El detector realiza la convolucion de un filtro sobre una imagen de 
            #caracteristicas HOG. Esta funcion permite que el algoritmo de ML
            #aprenda un filtro separable. Un 0 desactiva esta opcion y cualquier
            #otro valaor la activa.
            self.options.nuclear_norm_regularization_strength = 0
            #Esta opcion recibe como parametro el numero de hilos de ejecucion.
            #Para mayor velocidad de entrenamiento se ocupa el # de nucleos del CPU
            self.options.num_threads = 4
            #Se aumenta el numero de muestras de imaagenes si es necesario.
            #esta opcion recibe el numero de veces mximo que se debe aumentar.
            #Un mayor numero, incrementa el uso de memoria. NO se recomienda valores
            #superiores a 2 (default)
            self.options.upsample_limit = 1

        #Se carga el detector entrenado (para prueba)
        if loadPath is not None:
            self._detector = dlib.simple_object_detector(loadPath)
    #Fin del constructor
    ###################################################################

    # Funcion que preprocesa las ventanas que encierran al objeto
    def _prepare_annotations(self,annotations):
        annots = []
        for (x,y,xb,yb) in annotations:
            annots.append([dlib.rectangle(left=int(x),top=int(y),right=int(xb),bottom=int(yb))])
        return annots

    # Funcion que preprocesa las imagenes
    def _prepare_images(self,imagePaths):
        images = []
        #las imagenes se convierten a RGB que es el formato que acepta la funcion de entrenamiento
        for imPath in imagePaths:
            image = cv2.imread(imPath)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            images.append(image)
        return images

    def fit(self, imagePaths, annotations, trainAnnot, trainImages, visualize=False, savePath=None):
        st = time.time()
        print(st)
        annotations = self._prepare_annotations(annotations)
        images = self._prepare_images(imagePaths)
        ###############################################
        ############### Entrenamiento
        self._detector = dlib.train_simple_object_detector(images, annotations, self.options)
        print('Entrenamiento completado. Tiempo: {:.2f} segundos'.format(time.time() - st))
        #visualize HOG
        if visualize:
            win = dlib.image_window()
            win.set_image(self._detector)
            dlib.hit_enter_to_continue()
        #Se guarda el detector y se evalua su desempeño usando las imagenes de prueba
        if savePath is not None:
            self._detector.save(savePath)
            trainAnnot = self._prepare_annotations(trainAnnot)
            trainImages = self._prepare_images(trainImages)
            print("Metricas de entrenamiento : {}".format(dlib.test_simple_object_detector(trainImages,trainAnnot,self._detector)) )
        
        return self

    def predict(self,image):
    	#Se realiza la deteccion sobre un frame del video
        boxes = self._detector(image)
        preds = []
        for box in boxes:
            (x,y,xb,yb) = [box.left(),box.top(),box.right(),box.bottom()]
            preds.append((x,y,xb,yb))

        return preds
    def detect(self,image,annotate=None):
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        preds = self.predict(image)

        for (x,y,xb,yb) in preds:
            image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            #Se dibuja un contorno sobre el objeto detectado
            cv2.rectangle(image,(x,y),(xb,yb),(0,0,255),2)
            if annotate is not None and type(annotate)==str:
                cv2.putText(image,annotate,(x+5,y-5),cv2.FONT_HERSHEY_SIMPLEX,1.0,(128,255,0),2)          
                
        cv2.imshow("Detection",image)
        cv2.waitKey(1)
