# python-seg2mesh
Simple utility for extracting a single (or multiple) label from a h5 segmentation into a ply mesh.

## Requirement
- python 3.6
- numpy
- skimage
- h5py
- plyfile

## Usage 
Usage example:
```bash
$ python label_to_ply.py --path *path to segmentation file*.h5 --labels 10 34 101 
```
This script will create a .ply file for each label at the location of the segmentation file.

#### optional arguments
* save-path: path to alternative directory where to save ply file.
* center-origin: Default False. If true translate the object at the axis origin.


