# Facial-Recognition-System
Prerequisites-
1.Install python
2.Install keras with Tensorflow backend
3.Install libsvm tool


It contains the following:

1. Final report in pdf format

2. Three systems proposed.

	1. System 1

	- google inception model code in the name retrain.py which helps us to convert the image into representation 		vectors,have to pass the paths of image_dir and bottleneck_dir as arguments 
	- code to convert representation vectors in libsvm format.
	- example file for representation vector of size 1 x 2048.
	
	2. System 2
	
	- a single zip file containing all the codes and cnn.py contains the cnn that we built with many layers to classify 		the faces
	- dataset is present in three different folders train,test,validation

	 3. System 3
	 
	 - util folder contains the align-dlib python code.the aligned faces are stored in aligned-images directory
	 - batch-represent folder consists of lua scripts used to extract features from the model.
	 - generated-embeddings folder contains the extracted features
	 - the folder also contains code to convert the feature vectors into libsvm format
	 - for reference follow the face-recog-openface-steps.txt   

4. references used in our project (bibliography).
