
# Satellite Segmentation CNN Training and Implementation
![Header Image](https://github.com/xtinacarty/satellite-segmentation-final/blob/ea89db1107bad62c509251494002f84855bb86e9/misc%20assets/ReadME%20header.jpg)

Final project for Application Development, Summer Semester 2023. This application takes users from start to finish in the task of creating a Semantic Segmentation CNN and applying it on satellite imagery of tea plantations in Kenya. The application is divided into two parts corresponding with the two different platforms best suited for their respective tasks: Model creation and training in Python, and model implementation on the Kenya imagery in eCognition.

## Part 1: Model Training in Python
The first facet of the application is a guided, interactive python interface to preprocess data and train a CNN in a notebook environment. To make the application more user friendly, the actual code was seperated into the segmentation_model.py file in the form of several functions. In this way, users are not unnecessarily overwhelmed by all of the "under the hood" mechanics of the process, and instead are updated as to where in the process the code is at by several nested print statements within the segmentation_model.py functions. Additionally, the segmentation_model.py model was inundated with several moments of user-interaction that allows the user to confirm their personal set up re: data directories, customize the CNN, and visualize data at different points. The only code the user is exposed to are two functions pulled from segmentation_model.py; the *run_preprocessing()* function, which handles all of the data acquistion, preparation and preprocessing, and the *execute_model()* function, which handles all of the model creation and training. Below provides more detail as to the innerworkings of each function.

### Set Up and Data Pre-Processing - run_preprocessing()

This function begins by verifying the root directory of the user. The directory is automatically identified using the getcwd() command from the OS package and adding ‘\data’ to the path to infer the path to the folder containing the training data. Users also have the option to manually input the data folder path if the one presented from the command is incorrect.

Once the correct data path is set and the files are verified to be in the correct format, the images and their corresponding masks are divided into patches of 256 by 256 pixels to create more training data that is more easily digestible by the model. The patches are then converted into numpy arrays, the necessary format for CNN input data. The output of the patching should result into image and mask datasets that are the same length. If this is not the case, the program exits.

Next, colors are assigned to all six class labels which are then assigned to the labels delineated in each mask image. At this point, users are able to visualize the images and their respective masks to get a sense of what the training images and labels look like. At this point pre-processing is complete and users can move on to the next part of the code.


### Model Training, Execution and Saving - execute_model()


To begin, the data is split into training (75%) and testing (15%). The UNet model is then built from scratch layer by layer and called into a variable called “model”. A loss function is manually defined by combining dice and focal loss metrics. The model is compiled using an ADAM optimizer and users have the option to preview the model architecture and summary.
 
The customizable parameters in this section are the batch size and the training epochs. For batch size, if customization is desired, users are asked to choose from a list of options as batch size is limited to a power of 2 in order to ensure successful training. If users do not choose a value from the list presented, they are forced to re-enter until an acceptable value is received. For training epochs, there is no bound on the value inputted although users are advised on what types of values to choose to avoid lengthy training times while also avoiding overfitting issues related to too many training epochs.
At this point users are prompted to ensure they are ready to train the model and training begins. If users specify no at this point, then all progress from the execute_model() cell is lost and they must start over from the training process (all pre-processing progress is saved as long as the terminal remains open and active).  Once training is complete, users can preview the visual performance of the freshly trained model on unseen testing data as many times as desired. 



## Part 2: Model Implementation in eCognition
The second facet of the application is performed in the software ecognition, it integrates the model trained in the Part 1 to apply the Semantic Segmentation CNN in any image. The integration has been saved in *.dcp* file, format for ecognition. The *.dcp* contain a ruleset with the pre-processing to make compatible the images for the CNN execution module.
![Header Image](https://github.com/xtinacarty/satellite-segmentation-final/blob/main/misc%20assets/ReadME%20rulset%20tree%20ecog.jpg)

### Set Up and Image Pre-Processing - Preprocessing
The Preprocessing module transform the input image to a maximum pixel value of 255 and a 32Bit float image. This process needs to be done for the red band, blue band and green band. First of all is loaded the image to ecognition and the CNN_Rulsed.dcp file in to the process tree. Next is necessary to edit every process updating the output expression with the layer or mean layer of the image introduced, and the second is to select in the filter layer  the layer that corresponds to the channel corresponding to the name in the output layer (blue float, red float or green float). Finally, once all the processes are changed execute Preprocessing and it will create the new layers with the corresponding names assigned in the output layer.

![Header Image](https://github.com/xtinacarty/satellite-segmentation-final/blob/main/misc%20assets/ReadME%20Preprocessing%20ecog.jpg)
### Model Execution - CNN
The CNN module executes the Semantic Segmentation CNN the output is a vector layer object level with the classification ready to do the post processing that you wish.  It is just needed to link the model folder created in the Part 1 inside the process (load convolutional neuronal network) on the SavedModelpath insert the path location of your model.  After this the module CNN is ready to run and users are ready to do their own analysis.  
![Header Image](https://github.com/xtinacarty/satellite-segmentation-final/blob/main/misc%20assets/ReadME%20CNN%20ecog.jpg)
