The original data is taken from https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification. 

From there, I want to use the spectrogramCode.py to create the original spectrogram which will be used to extract the numerical features in the future.
Then we will use the splittingData.py to create a training and testing dataset, ensuring the spectogram and wav files match. From there rest of the files 
extract all the numerical features and the stft spectrogram which will be used for the CNN model.

Currently, there is a CNN model file where the two different spectograms are loaded and treated as two branches which merge into one. The DNN file created a 
neural network with all the numerical features. From here, I will treat the DNN model as another branch and merge that within the CNN model as well, and the 
goal is for this to effectively classify music genres. 
