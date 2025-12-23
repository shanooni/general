import numpy as np
import tensorflow as tf
from img2vec_pytorch import Img2Vec
from PIL import Image
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
class Processor:
    def __init__(self):
        self.IMG_SIZE = 224
        self.PIXEL_VALUES = 255
        self.HEIGHT = 16
        self.WIDTH = 16
        self.CHANNELS = 2
        self.img_vec = Img2Vec(model = "resnet18")
    
        
    def process_image(self, image: Image.Image):
        image = image.resize((self.IMG_SIZE, self.IMG_SIZE))
        image = np.array(image)
        image = image / self.PIXEL_VALUES
        return image
    
    def process_image_cnn(self, images):
        feature_vec = self.img_vec.get_vec(images)
        feature_vec = np.array(feature_vec)
        feature_vec = feature_vec.reshape((1, self.HEIGHT, self.WIDTH, self.CHANNELS))
        return feature_vec
    
    def scale_feature(self, images):
        feature_vec = self.img_vec.get_vec(images)
        feature_vec = feature_vec.reshape(1, -1)
        scaled_feature_vector = scaler.fit_transform(feature_vec)
        return scaled_feature_vector