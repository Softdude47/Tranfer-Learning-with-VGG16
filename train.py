# importing libraries
print("[INFO]: import Libraries")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

import h5py
import argparse

# commandline args
print("[INFO]: setting commandline params")
ap = argparse.ArgumentParser()
ap.add_argument("--input_path", "-ip", help="path to dataset", type=str, required=True)
ap.add_argument("--n_jobs", "-jobs", help="number of cpu cores to use in training ", type=int, default=-1)
ap.add_argument("--training_size", "-ts", help="training size(in percentage) of dataset", type=int, default=0.7)
ap.add_argument("--show_info", "-show", help="specify wether to showing training info", type=bool, default=True)
args = vars(ap.parse_args())

show = args["show_info"]
jobs = args["n_jobs"]
db_path = args["input_path"]
train_size = args["training_size"]

# database
print("[INFO]: opening Database")
db = h5py.File(name=db_path, mdoe="r")
features = db["features"]
labels = db["labels"]
class_name = db["class_names"]

# computing trianing index
index = len(features) * train_size
index = int(index)

# tunnable parameters
print("[INFO]:  training")
param = {"C": [1.0, 10.0, 100.0, 1000, 10000]}

# passing parameter to a helper function and training it
model = GridSearchCV(estimator=LogisticRegression(), n_jobs=jobs, param_grid=param)
model.fit(features[ : index], labels[ : index])

# testing
print("[INFO]: predicting")
predictions = model.predict(features[index : ])

report = classification_report(labels[index : ], predictions, target_class=class_name)
print(report)