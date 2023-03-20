'''
Use ResNet50, a 50-layer CNN model pretrained on > 40M images

Following tutorial from https://campus.datacamp.com/courses/introduction-to-deep-learning-with-keras/advanced-model-architectures?ex=6
'''
import numpy as np
# import image from keras preprocessing
from tensorflow.keras.preprocessing import image

# tensorflow keras applications resnet50
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50, decode_predictions

img_path = '../data/img.png'
# load the image with the right target size for the model
img = image.load_img(img_path, target_size=(224,224))

# turn image into array
img = image.img_to_array(img)

# expand the dimensions so that it's understood by the network:
# img.shape turns from (224,224,3) into (1,224,224,3)
img = np.expand_dims(img, axis=0)

# pre-process the img in the same way training images were
img = preprocess_input(img)

# Instantiate a ResNet50 model with imagenet weights
model = ResNet50(weights ='imagenet')

# predict with ResNet50 on our img
preds = model.predict(img)

# Decode predictions and print it
print('Predicted:', decode_predictions(preds, tp=1)[0])