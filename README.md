# ML-TEAM-BLACK

Computer vision challenge: object detection in images
<br>

Challenge goals are:
- Identify the items listed below explicitly in any image.
- A bonus goal would be to determine the number of each of these items in a picture.
- Final super-bonus objective would be identifying other items outside of the list and reporting back.
<br>

The objects to identify are:
- Phone
- Laptop
- Satellite Dish
- USB stick
- Keyboard
- Router
- Keys
- Magnifying Glass
- Server rack
- Mouse

Image data on 10 object classes collected by [Natalia](https://github.com/natalijamahoby). The dataset can be [found on Kaggle](https://www.kaggle.com/datasets/ksenia5/electronic-object-detection).
<br>

We used the pretrained convolutional neural network InceptionV3 with the Keras api, to retrain the model on our custom dataset for categorical image classification with transfer learning.
<br>

Images were split in 70-20-5 ratio for train-test-split and resized to 299x299 (RGB) for input to model, as this was the original size model was trained on. Data augmentations (rotations, shears, scaling, ...) were used to increase dataset for training. After 10 epochs of training an accuary of 83% was achieved.

<br>

Regularization with drop out of 20% was used to improve the prediction results by preventing overfitting.

<br>

Object detection with transfer learning using pretrained InceptionV3 convolutional neural network [first attempt here](https://www.kaggle.com/code/ksenia5/transfer-learning-with-inception) and a [better performing model here](https://www.kaggle.com/ksenia5/transfer-learning-with-inceptionv3).
