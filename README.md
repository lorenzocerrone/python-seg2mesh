# python-seg2mesh
Simple utility for exporting single or multiple labels from a segmentation (h5 file only) into a mesh (ply only).

## Requirement
- python 3.6 or 3.7
- numpy
- scikit-image
- h5py
- plyfile

# Installation
If you are using anaconda python:
```bash
$ conda install -c anaconda numpy h5py
$ conda install -c conda-forge scikit-image
$ pip install plyfile
```


## Usage 
Usage example:
```bash
$ python label_to_ply.py --path *path to segmentation file*.h5 --labels 10 34 101 
```
This script will create a .ply file with all objects in the same file.

#### optional arguments
* multi-file: If "True" all meshes are saved in a different file. 
* save-path: path to alternative directory where to save ply file.
* center-origin: Default False. If true translate the object at the axis origin.
* dataset: Default "label". Name of the h5 dataset to retrieve the labels from.

