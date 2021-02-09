from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

import h5py
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--input_path", "-ip", help="path to dataset", type=str, required=True)
ap.add_argument("--training_size", "-ts", help="training size(in percentage) of dataset", type=str, default=0.7)
ap.add_argument("--showing_info", "-show", help="specify wether to showing training info", type=bool, default=True)
args = vars(ap.parse_args())

db_path = args["ip"]
train_size = args["ts"]
show = args["show"]

db = h5py.File(name=db_path, mdoe="r")
features = db["features"]
labels = db["labels"]
class_name = db["class_names"]

index = len(features) * train_size
index = int(index)

param = {"C": [1.0, 10.0, 100.0, 1000, 10000]}
model = GridSearchCV(estimator=LogisticRegression(), param_grid=param)
model.fit(features[ : index], labels[ : index])

predictions = model.predict(features[index : ])

report = classification_report(labels[index : ], predictions, target_class=class_name)