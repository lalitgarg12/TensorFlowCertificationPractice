# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# try:
#     # %tensorflow_version only exists in Colab.
#     %tensorflow_version 2.x
#     !pip install -q -U tfx==0.21.2
#     print("You can safely ignore the package incompatibility errors.")
# except Exception:
#     pass

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "data"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# %%writefile person.proto
# syntax = "proto3";
# message Person {
#   string name = 1;
#   int32 id = 2;
#   repeated string email = 3;
# }
#
# !protoc person.proto --python_out=. --descriptor_set_out=person.desc --include_imports
#
# !ls person*
#
# from person_pb2 import Person
#
# person = Person(name="Al", id=123, email=["a@b.com"])  # create a Person
# print(person)  # display the Person
#
# person.name  # read a field
#
# person.name = "Alice"  # modify a field
#
# person.email[0]  # repeated fields can be accessed like arrays
#
# person.email.append("c@d.com")  # add an email address
#
# s = person.SerializeToString()  # serialize to a byte string
# s
#
# person2 = Person()  # create a new Person
# person2.ParseFromString(s)  # parse the byte string (27 bytes)
#
# person == person2  # now they are equal
#
# person_tf = tf.io.decode_proto(
#     bytes=s,
#     message_type="Person",
#     field_names=["name", "id", "email"],
#     output_types=[tf.string, tf.int32, tf.string],
#     descriptor_source="person.desc")
#
# person_tf.values
#
#from tensorflow.train import BytesList, FloatList, Int64List
#from tensorflow.train import Feature, Features, Example
BytesList = tf.train.BytesList
FloatList = tf.train.FloatList
Int64List = tf.train.Int64List
Feature = tf.train.Feature
Features = tf.train.Features
Example = tf.train.Example

person_example = Example(
    features=Features(
        feature={
            "name": Feature(bytes_list=BytesList(value=[b"Alice"])),
            "id": Feature(int64_list=Int64List(value=[123])),
            "emails": Feature(bytes_list=BytesList(value=[b"a@b.com", b"c@d.com"]))
        }))

with tf.io.TFRecordWriter("my_contacts.tfrecord") as f:
    f.write(person_example.SerializeToString())

feature_description = {
    "name": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "id": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "emails": tf.io.VarLenFeature(tf.string),
}
for serialized_example in tf.data.TFRecordDataset(["my_contacts.tfrecord"]):
    parsed_example = tf.io.parse_single_example(serialized_example,
                                                feature_description)

print(parsed_example)

print(parsed_example["emails"].values[0])

print(tf.sparse.to_dense(parsed_example["emails"], default_value=b""))

print(parsed_example["emails"].values)

