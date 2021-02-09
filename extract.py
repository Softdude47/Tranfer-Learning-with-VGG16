print("[INFO]: loading Libraries")
from tensorflow.keras.applications import VGG16
from sklearn.preprocessing import LabelEncoder
from imutils.paths import list_images
from os.path import sep as separator
from random import shuffle

from datasets.simple_dataset_loader import Simple_Dataset_Loader
from preprocessors.image_to_array import Image_to_Array
from preprocessors.imagenet import Imagenet
from cacher.file_cacher import File_Database

import argparse

# commandline parameters
ap = argparse.ArgumentParser()
ap.add_argument("--input_path", "-ip", help="path to dataset", type=str, required=True)
ap.add_argument("--output_path", "-op", help="path to dataset", type=str, required=True)
ap.add_argument("--show_info", "-show", help="show info on extraction process", type=bool, default=True)
ap.add_argument("--image_dimension", "-dim", help="target dimension of each images in the dataset\nfor example (480, 640)", type=tuple, default=(224,224))
ap.add_argument("--batch_size", "-bs", help="batch size", type=int, default=32)
ap.add_argument("--buffer_size", "-bf", help="buffer size", type=int, default=1000)
args = vars(ap.parse_args())

bs = args["bs"]
buffer_size = args["bf"]
input_path = args["ip"]
output_path = args["op"]
target_size = args["dim"]
show_info = args["show"]

# setup image dimension
image_paths = list(list_images(input_path))
dimension = [len(image_paths),]
dimension.extend(target_size)

print("[INFO]: initializing Key functions")
# initialize the feature extractor(VGG16) model
feature_extractor = VGG16(include_top=False, weights="imagenets")

# initialize and create list of preprocessors
IAp = Image_to_Array()
Ip = Imagenet()
preprocessors = [IAp, Ip]
sdl = Simple_Dataset_Loader(preprocessors=preprocessors)

# initialize database
print("[INFO]: creating Database")
db = File_Database(output_path=output_path, buffSize=buffer_size, dimension=dimension)

# extract image labels(from image paths) and fit it into encoder
class_names = [i.split(separator)[-2] for i in image_paths]
le = LabelEncoder().fit(class_names)

# store string format of image labels(from image paths) to database
db.store_class_labels(le.classes_)

# loop over image paths in batches
print("[INFO]: extracting feature")
for i in range(0, dimension[0], bs):
    # load images and extract their labels
    batchPath = image_paths[i : i + bs]
    batchImages, batchLabels = sdl.preprocess(batchPath, target_size=target_size, include_label=True)
    
    # encode label and extract feautres
    batchLabels = le.transform(batchLabels)
    batchImages = feature_extractor.predict(batchImages)
    
    # adds extracted features and encodd labels to database
    db.add(batchImages, batchLabels)
    if show_info:
        print(f"[INFO]: process {i}/{dimension}")

# close database
db.close()
print("[INFO]: success....")