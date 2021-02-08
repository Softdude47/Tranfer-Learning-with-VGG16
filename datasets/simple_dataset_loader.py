

from keras.preprocessing.image import load_img


class Simple_Dataset_Loader:
    def __init__(self, preprocessors=None):
        "takes in list of Preprocessor Classes"
        self.preprocessors = preprocessors
        if self.preprocessors is None:
            self.preprocessors = []
            
    def preprocess(self, img_path: str, target_size=None, include_labels : bool = True)-> list or tuple:
        """ preprocess the images with the Procoessors """
        img_path_list = list(img_path)
        loaded_img = []
        
        # loops over the image paths/urls
        for paths in img_path_list:
            
            # loads the images
            image = self.load_image(paths, target_size=target_size)
            
            # preprocess the images with the Preprocessor Classes
            for preprocessor in self.preprocessors:
                image = preprocessor.preprocess(image)
                
            # append the preprocess image to the list
            loaded_img.append(image)
        
        # get the image labels if the "include_labels" parameter is set to true
        if include_labels:
            image_labels = [i.split("/")[-2] for i in img_path]
            return (loaded_img, image_labels)
        return loaded_img
    
    def load_img(self, image, target_size=None, grayscale=False, interpolation="nearest"):
        loaded_img = load_img(image, grayscale=grayscale, target_size=target_size, interpolation=interpolation)
        return loaded_img