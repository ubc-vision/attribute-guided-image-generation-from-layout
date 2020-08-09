import numpy as np
import h5py
import json
import torch

# load matrix.pt
# m = torch.load("data/matrix_obj_vs_att.pt")
# print("occurrences of tree and green: ", m[2, 3])
# print("occurrences of sky and blue: ", m[11, 2])
# print(m.shape)

data = {}
with h5py.File("data/train.h5", 'r') as f:
    for k, v in f.items():
        if k == 'image_paths':
            image_paths = list(v)
        else:
            data[k] = torch.IntTensor(np.asarray(v))

with open("~/data/vocab.json", 'r') as f:
    vocab = json.load(f)

num_objects = len(vocab['object_idx_to_name'])
num_attributes = len(vocab['attribute_idx_to_name'])

object_array = data['object_names']
attribute_array = data['object_attributes']

print("number of objects: ", num_objects)
print("number of attributs: ", num_attributes)
print("train object shape: ", object_array.shape)
print("train attributes shape: ", attribute_array.shape)

print(vocab['object_idx_to_name'])
print(vocab['attribute_idx_to_name'])

matrix_obj_vs_att = torch.zeros(num_objects, num_attributes)

for i in range(object_array.shape[0]):  # for each image
    for j in range(object_array.shape[1]):  # for each object
        if object_array[i, j] == -1:
            break   # done image i
        obj = object_array[i, j]
        for k in range(attribute_array.shape[2]):
            if attribute_array[i, j, k] == -1:
                break # done object j
            att = attribute_array[i, j, k]
            matrix_obj_vs_att[obj, att] += 1

print("occurrences of tree and green: ", matrix_obj_vs_att[2, 90])
print("occurrences of sky and blue: ", matrix_obj_vs_att[11, 94])

# print("occurrences of tree and green: ", matrix_obj_vs_att[2, 3])
# print("occurrences of sky and blue: ", matrix_obj_vs_att[11, 2])

torch.save(matrix_obj_vs_att, "data/matrix_obj_vs_att.pt")
