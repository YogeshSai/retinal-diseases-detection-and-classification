# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

# Load the pre-trained model
model = load_model('model1.h5')

# Load an OCT image to classify
img = cv2.imread('image4.jpeg')
img = cv2.resize(img, (224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# Predict the class of the image
preds = model.predict(x)
if preds[0][0] > 0.5:
    text = "CNV"
elif preds[0][1] > 0.5:
    text = "DME"
elif preds[0][2] > 0.5:
    text = "DRUSEN"
else:
    text = "NORMAL"

# Add the predicted class text to the image
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, text, (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

# Show the image
cv2.imshow('OCT Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


