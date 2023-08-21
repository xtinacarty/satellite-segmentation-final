# Satellite Segmentation CNN Training and Implementation
Final project for Application Development, Summer Semester 2023. This application takes users from start to finish in the task of creating a Semantic Segmentation CNN and applying it on satellite imagery of tea plantations in Kenya. The application is divided into two parts corresponding with the two different platforms best suited for their respective tasks: Model creation and training in Python, and model implementation on the Kenya imagery in eCognition.

## Part 1: Model Training in Python
The first facet of the application is a guided, interactive python interface to preprocess data and train a CNN in a notebook environment. To make the application more user friendly, the actual code was seperated into the segmentation_model.py file in the form of several functions. In this way, users are not unnecessarily overwhelmed by all of the "under the hood" mechanics of the process, and instead are updated as to where in the process the code is at by several nested print statements within the segmentation_model.py functions. Additionally, the segmentation_model.py model was inundated with several moments of user-interaction that allows the user to confirm their personal set up re: data directories, customize the CNN, and visualize data at different points. The only code the user is exposed to are two functions pulled from segmentation_model.py; the *run_preprocessing()* function, which handles all of the data acquistion, preparation and preprocessing, and the *execute_model()* function, which handles all of the model creation and training. Below provides more detail as to the innerworkings of each function.

### run_preprocessing()

### execute_model()

## Part 2: Model Implementation in eCognition
