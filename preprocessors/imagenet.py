from tensorflow.keras.applications import imagenet_utils

class Imagenet:
    def __init__(self) -> None:
        pass
    
    def preprocess(self, image):
        preprocessed_image = imagenet_utils.preprocess_input(image)
        return preprocessed_image