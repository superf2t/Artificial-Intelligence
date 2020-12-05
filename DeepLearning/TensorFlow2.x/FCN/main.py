#model
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,  Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import Dropout, Input
from tensorflow.keras.initializers import Constant

image_shape = (160, 576)
