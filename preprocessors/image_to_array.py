from tensorflow.keras.preprocessing.image import img_to_array

class Image_to_Array:
    def __init__(self,):
        pass
    def preprocess(self, image):
        image = img_to_array(image)
        return image