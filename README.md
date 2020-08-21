# seg2mesh

Simple python utility for exporting single or multiple labels from a segmentation (`h5` file only) into a mesh (`ply` only).

## Requirements

- python 3.7
- numpy
- scikit-image
- h5py
- vtk
- ilastik/marching_cubes
- ray

## Installation

If you are using anaconda python:

```bash
conda install -c conda-forge scikit-image h5py numpy vtk netcdf4
conda install -c ilastik-forge -c conda-forge marching_cubes
pip install ray
```

## Versions

- **0.4 beta - Multiprocessing (work in progress...)**
  - Use the `--multiprocessing` flag to enable parallel processing of labels using all available cores.
  - Internally seg2mesh uses `ray` which should allow delployment on a cluster.
  - Had to revert to the `skimage` marching_cubes implementation insteads of `ilastik`.
  - (!) although running, multiprocessing is **not** currently faster. Code must still be adapted

- **0.3 - Code revamp for performances**
  - use `ilastik` marching_cubes implementation insteads of `skimage`, speed gain: ~2x.
  - finding the largest connected component is now ~2x faster.
  - streamlined of decimation and smoothing that are now directly applied to the mesh (no more saving to disk in between).
  - use of `vtk` for exporting `PLY` files.
  - labels below a `--min-volume` thhreshold are not processed
  - performances gain: ~90% faster than `v0.2`:

```bash
# labels_2_ply (v0.2)
python labels_to_ply.py --path test_files/fused_dc_cropped--C00--T00020_crop_x40-1630_y520-970_predictions.h5 --dataset "merged_50000" --labels 42 69 3 88 67 --simple-name "test" --reduction 0.9
(...)
Finished!
5 objects in 0:01:46.949603 [HH:mm:ss]
```

```bash
# seg2mesh (v0.3)
python seg2mesh.py --path test_files/fused_dc_cropped--C00--T00020_crop_x40-1630_y520-970_predictions.h5 --dataset "merged_50000" --labels 42 69 3 88 67 --out-name "test" --reduction 0.9
(...)
Finished!
5 objects in 0:01:07.503175 [HH:mm:ss]
```

- **0.2 - Flexible output and batch mode**
  - path and name of the output file can be specified
  - getting all the labels in a file with `--all`
  - implemented batch modes:
    - `batch`: specific labels from specific files
    - `batch-all`: all labels from all files

- **0.1 - Initial release**

## Usage

To extract specific labels from a file, generating a .ply file  for each label:

```bash
python seg2mesh.py --path *path to segmentation file*.h5 --dataset *name of dataset containing labels in h5 file- --labels 10 34 101
```

To extract all labels from a file, store the .ply files in a specific folders and name them _foo_xxx.ply_:

```bash
python seg2mesh.py --path *path to segmentation file*.h5 --dataset *name of dataset containing labels in h5 file- --all --out-path *path to output folder- --out-name "foo"
```

### Mandatory arguments

- path: Path to the .h5 file to process.
- dataset: Default "label". Name of the h5 dataset to retrieve the labels from (all dataset can be listed by `h5ls *path to segmentation file*.h5`).

### Optional arguments

- **labels**: list of labels to extract (space separated), example= `--labels 12 42 33 47`.
- **all**: if passed all labels of the file are extracted.
- **out-path**: path to directory where to save ply file.
- **out-name**: use this as base name for output file(s).
- **min-volume**: minimal volume of label to be extracted (in voxels).
- **batch**: tab-delimited file containing list of time points and labels to process (see *Batch mode- below)
- **batch-all**: the script will extract all labels from all files which names are similar to the input files (i.e. all _t/Txxxxx_ time points)
- **multiprocessing**: if called enables parallel processing of labels using all available cores.

### Filters arguments

These modify the mesh before it is saved

- **reduction**: If reduction > 0 a decimation filter is applied. MaxValue 1.0 (100% reduction).
- **smoothing**: If called a Laplacian smoothing filter is applied.

#### Batch mode

This allows the processing of several `.h5` files.
Two modes are implemented:

- **1. Specific labels from specific files**:

```bash
python seg2mesh.py --path *path to one of the segmentation file*.h5 --dataset *name of dataset containing labels in h5 file- --batch *path to tab delimited file with list of time points and labels*
```

The script will iterate over all time points listed in the batch file and for each, generate .ply files for all labels for that time. The  *.ply- files are automatically sorted in subfolders (*tXX*)

The batch file should be tab-seprarated and contain two columns:

- 1st column: the time point to process encoded on *two digits*
- 2nd column: the labels to extract separated by a space

Example:

```bash
Frame   Labels
02      279 256
06      258 42 10 11
```

- **2. Extract all labels from all files**

The files must contain a time point stamp of format `T/tXXXXX` (_ex._ `t00012`). The script will parse all files fitting this pattern and arrange the `.ply` files in subfolders (`tXX`).

Example:

```bash
python seg2mesh.py --path *path to segmentation file*.h5 --dataset *name of dataset containing labels in h5 file- --batch-all --out-path *path to output folder- --out-name "foo"
```
