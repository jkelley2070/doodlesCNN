# Repository Overview
## CNN_models.py  
  - Contains the classes containing all 9 cnn models
    - CNN1 - A simple convolutional neural network model.(LENET)
    - CNN2 - An improved convolutional neural network model with additional layers and modifications.
    - CNN3 - A version with more filters than the base model.
    - CNN4 - A version with additional dense layers.
    - CNN5 - A version with more filters.
    - CNN6 - A version with double convolutional blocks.
    - bestCNN - A version with hyperparameter optimization using Ray Tune.
    - CNN8 - A version with residual blocks based on the Dive into Deep Learning textbook.
    - CNN9 - A version with more layers of residual blocks.
  - Used in Train_Models and Test_Models
## Labels.py
 - Labels for each class
## MLP_models.py
  - Contains the class for all 6 MLPS
    - **MLP1**: Base model, simple multilayer perceptron. (mnist)
    - **MLP2**: Increases the number of neurons per layer.
    - **MLP3**: Adds an additional layer with more neurons.
    - **MLP4**: Extends MLP3 by adding even more neurons and a layer.
    - **MLP5**: Introduces dropout and batch normalization.
    - **bestMLP**: Optimized using Ray Tune for best performance.
## TensorboardLogs1.csv
  - Contains all the tracked progression of the models throughout all epochs!
## Testing_Models.ipynb
  - Jupytner notebook for running the pretrained models
  - Can be viewed at the following colab [Link](https://colab.research.google.com/drive/1zkpwkQtjcSRcWSczfdfOQRqlynhShQ8-?usp=sharing)
## Train_Models.ipynb
  - Jupytner notebook for training the models
## final_testing.csv
  - Statistical result the final testing evaluations on the best CNN
