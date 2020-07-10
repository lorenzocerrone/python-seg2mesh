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
Usage examples:

To extract specific labels from a file, generating a .ply file  for each label:
```bash
$ python label_to_ply.py --path *path to segmentation file*.h5 --dataset *name of dataset containing labels in h5 file* --labels 10 34 101
```
To extract all labels from a file, store the .ply files in a specific folders and name them _foo_xxx.ply_:

```bash
$ python label_to_ply.py --path *path to segmentation file*.h5 --dataset *name of dataset containing labels in h5 file* --all --save-path *path to output folder* --simple-name "foo"
```

#### mandatory arguments
* path: Path to the .h5 file to process.
* dataset: Default "label". Name of the h5 dataset to retrieve the labels from.

#### optional arguments
* labels: list of labels to extract
* all: if passed all labels of the file are extracted.
* single-file: if passed, all meshes are saved in the same file.
* save-path: path to directory where to save ply file.
* center-origin: default False. If true translate the object at the axis origin.
* step-size: default 1. Marching cube step size (only int). The higher the step size the coarser the output.
* simple-name: use this as base name for output file(s).
* batch: tab-delimited file containing list of time points and labels to process (see Batch mode below)
* batch-all: the script will extract all labels from all files similar to the input files (i.e. all _t/Txxxxx_ time points)

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
Usage examples:
```bash
$ python label_to_ply.py --path *path to one of the segmentation file*.h5 --dataset *name of dataset containing labels in h5 file* --batch *path to tab delimited file with list of time points and labels*
```
This script will iterate over all time points listed in the batch file and for each generate .ply files for all labels. Output .ply files are automatically sorted in subfolders (tXX)

Formating of batch file
```bash
Frame	labels
02	279 256
06	258 42 10 11
```
To extract all labels from all time points, store the .ply files in a specific folders and name them _foo_xxx.ply_:

```bash
$ python label_to_ply.py --path *path to segmentation file*.h5 --dataset *name of dataset containing labels in h5 file* --batch-all --save-path *path to output folder* --simple-name "foo"
```
This scripts analyses the file name and expect a time point stamp of format _T/t00012_, if so it will parse all files fitting this pattern and store the ply files in subfolders (tXX).