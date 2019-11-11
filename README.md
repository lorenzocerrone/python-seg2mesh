# python-seg2mesh

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
Simple usage example:
```bash
$ python label2ply.py --path *path to segmentation file*.h5 --labels 10 34 101 
```
This script will create a .ply file for each label at the location of the segmentation file.

#### optional arguments
* save-path: path to alternative directory where to save ply file.
* center-origin: Default False. If true translate the object at the axis origin.


