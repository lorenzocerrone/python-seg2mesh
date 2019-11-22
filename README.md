# python-seg2mesh
Simple utility for exporting single or multiple labels from a segmentation (h5 file only) into a mesh (ply only).

## Requirement
- python 3.6
- numpy
- scikit-image
- h5py
- plyfile
- vtk

# Installation
If you are using anaconda python:
```bash
$ conda install -c anaconda numpy h5py vtk
$ conda install -c conda-forge scikit-image
$ conda install -c vtk
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
* step-size: Default 1. Marching cube step size (only int). The higher the step size the coarser the output.

#### Filters arguments
The filters are implemented as "safe" operations. 
Thus they will not modify the main original mesh but create a new unique file.\
Usage example:
```bash
$ python label_to_ply.py --path *path to segmentation file*.h5 --labels 10 --reduction 0.25 --iterations 100
```
* reduction: If reduction > 0 a decimation filter is applied. MaxValue 1.0 (100%reduction).
* iterations: If iteration > 0 a Laplacian smoothing filter is applied.
* relaxation: The smaller the better accuracy but slower convergence. Default 0.1.
* edge-smoothing: Apply edge smoothing. Default False, seems to help after very intensive decimation.

#### Batch mode
Usage example:
```bash
$ python label_to_ply.py --path *path to one of the segmentation file*.h5 --batch *path to tab delimited file with list of time points and labels*
```
This script will iterate over all time points listed in the batch file and for each generate .ply files for all labels.

Formating of batch file
```bash
Frame	labels
02	279 256
06	258 42 10 11
```