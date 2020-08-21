import os
import csv
import re
import numpy as np
import time
import datetime
import argparse
import h5py
from pathlib import Path

import psutil
from concurrent import futures
import traceback

# TODO  replace skimage measure.label() and/or marching_cubes by vtk functions? [if exist and are faster...].
from skimage import measure
# This is incomaptible w/ the Ray multiprocessing library
from marching_cubes import march

from vtk import vtkPolyData, vtkCellArray, vtkPoints, vtkPolygon, vtkPLYWriter, vtkDecimatePro, vtkSmoothPolyDataFilter, vtkPolyDataNormals
from vtk.util import numpy_support

####
_version = "0.4 beta"
###

def ndarray2vtkMesh(inVertexArray, inFacesArray):
    ''' Code inspired by https://github.com/selaux/numpy2vtk '''
    # Handle the points & vertices:
    z_index=0
    vtk_points = vtkPoints()
    for p in inVertexArray:
        z_value = p[2] if inVertexArray.shape[1] == 3 else z_index
        vtk_points.InsertNextPoint([p[0], p[1], z_value])
    number_of_points = vtk_points.GetNumberOfPoints()

    indices = np.array(range(number_of_points), dtype=np.int)
    vtk_vertices = vtkCellArray()
    for v in indices:
        vtk_vertices.InsertNextCell(1)
        vtk_vertices.InsertCellPoint(v)

    # Handle faces
    number_of_polygons = inFacesArray.shape[0]
    poly_shape = inFacesArray.shape[1]
    vtk_polygons = vtkCellArray()
    for j in range(0, number_of_polygons):
        polygon = vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(poly_shape)
        for i in range(0, poly_shape):
            polygon.GetPointIds().SetId(i, inFacesArray[j, i])
        vtk_polygons.InsertNextCell(polygon)

    # Assemble the vtkPolyData from the points, vertices and faces
    poly_data = vtkPolyData()
    poly_data.SetPoints(vtk_points)
    poly_data.SetVerts(vtk_vertices)
    poly_data.SetPolys(vtk_polygons)

    return poly_data

def writePLYfile(vtkPoly, savepath = None):
    # write results to output file

    writer = vtkPLYWriter()
    writer.SetInputData(vtkPoly)
    writer.SetFileName(savepath)
    writer.Write()
    print(f" -File: {savepath}")
    return 1

def decimation(vtkPoly, reduction=0.25):

    # decimate and copy data
    decimate = vtkDecimatePro()
    decimate.SetInputData(vtkPoly)
    decimate.SetTargetReduction(reduction)  # (float) set 0 for no reduction and 1 for 100% reduction
    decimate.Update()

    decimatedPoly = vtkPolyData()
    decimatedPoly.ShallowCopy(decimate.GetOutput())

    return decimatedPoly

def smooth(vtkPoly, iterations=100, relaxation=0.1, edgesmoothing=True):
    # Smooth mesh with Laplacian Smoothing
    smooth = vtkSmoothPolyDataFilter()
    smooth.SetInputData(vtkPoly)
    smooth.SetRelaxationFactor(relaxation)
    smooth.SetNumberOfIterations(iterations)
    if edgesmoothing:
        smooth.FeatureEdgeSmoothingOn()
    else:
        smooth.FeatureEdgeSmoothingOff()
    smooth.BoundarySmoothingOn()
    smooth.Update()

    smoothPoly = vtkPolyData()
    smoothPoly.ShallowCopy(smooth.GetOutput())

    # Find mesh normals (Not sure why)
    normal = vtkPolyDataNormals()
    normal.SetInputData(smoothPoly)
    normal.ComputePointNormalsOn()
    normal.ComputeCellNormalsOn()
    normal.Update()

    normalPoly = vtkPolyData()
    normalPoly.ShallowCopy(normal.GetOutput())

    return normalPoly

def getLargestCC(segmentation):
    """Returns largest connected components"""
    # ~2x faster than clean_object(obj)
    print(" -Cleaning small detached objects...")
    t0 = time.time()

    # find bounding box
    _x, _y, _z = np.nonzero(segmentation)
    bb_segmentation = segmentation[_x.min():_x.max(),
                                   _y.min():_y.max(),
                                   _z.min():_z.max()]

    # relabel connected components
    labels = measure.label(bb_segmentation)

    assert(labels.max() != 0) # assume at least 1 CC
    _largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1

    # put bounding box back in the original volume
    largestCC = np.zeros(segmentation.shape, dtype=_largestCC.dtype)
    largestCC[_x.min():_x.max(), _y.min():_y.max(), _z.min():_z.max()] = _largestCC

    t1 = time.time()
    print(f"  [{round(t1-t0, 3)} secs]")
    return largestCC

def _getLargestCC(segmentation):
    """Legacy version, substitute by getLargestCC"""
    """Returns largest connected components"""
    # ~2x faster than clean_object(obj)
    print(" -Cleaning small detached objects...")
    t0 = time.time()

    # relabel connected components
    labels = measure.label(segmentation)
    assert(labels.max() != 0) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    t1 = time.time()
    print(f"  [{round(t1-t0, 3)} secs]")
    return largestCC

def get_label(segmentation, label, min_vol = 0):
    """Extract a mask where the label is"""
    print(f" -Extracting object...")
    obj = segmentation == label
    # Compute its volume
    volume = np.count_nonzero(obj)
    if volume < min_vol:
        print(f" -Below threshold: (min {min_vol}), skipping.")
        print(f"{25*'-'}")
        return None
    return obj

def label2vtk(segmentation, label, min_vol = 0):
    """Compute a mesh from a single object"""
    # Retrieve the segmentation corresponding to the label, keep it if > min_volume
    obj = get_label(segmentation, label, min_vol = min_vol)

    if obj is None:
        return None

    if not np.any(obj):
        # If no index match nothing to do
        print(f" -Label: {label} Not found")
        return None
    # Get the largest connected component
    obj = getLargestCC(obj)
    obj = obj.astype(float)


    print(f" -Mesh creation...")
    t0 = time.time()
    # Generate a mesh using the marching cubes algorithm.
    # ilastik/marching_cubes implementation is ~ 2x faster than Skimage's one
    # vertx, _, faces = march(obj.T, 0)
    # However ilastik is not compatible w/ Ray parallelisation
    vertx, faces, normals, _ = measure.marching_cubes(obj, 0, step_size=_step_size)
    # Convert the vertices and faces in a VTK polyData
    vtkPoly = ndarray2vtkMesh(vertx, faces.astype(int))
    print(f"  [{vtkPoly.GetNumberOfPoints()} vertices | {vtkPoly.GetNumberOfPolys()} faces]")
    t1 = time.time()
    print(f"  [{round(t1-t0, 3)} secs]")
    return vtkPoly

def get_all_labels(segmentation, dataset):
    print(" -Getting all labels")
    all_labels = np.unique(segmentation)
    print(f" -Found {len(all_labels)} labels\n{25*'-'}")
    return all_labels

def get_segmentation(path, dataset):
    print(f"{50*'='}\nProcessing:\n{path}")
    with h5py.File(path, "r") as f:
        segmentation = f[dataset][...]
    return segmentation

def label2mesh(segmentation, label, min_vol=0, save_path=None, outfile_basename="", save_subfolder=""):
    print(f" -Processing: {label}")
    vtkPoly = label2vtk(segmentation, label, min_vol = min_vol)
    if vtkPoly is not None:
        # Mesh decimation
        if _reduction > 0:
            print(" -Applying decimation...")
            t0 = time.time()
            vtkPoly = decimation(vtkPoly, reduction = _reduction)
            print(f"  [{vtkPoly.GetNumberOfPoints()} vertices | {vtkPoly.GetNumberOfPolys()} faces]")
            t1 = time.time()
            print(f"  [{round(t1-t0, 3)} secs]")

        if _smoothing:
            print(" -Applying smoothing...")
            t0 = time.time()
            vtkPoly = smooth(vtkPoly)
            print(f"  [{vtkPoly.GetNumberOfPoints()} vertices | {vtkPoly.GetNumberOfPolys()} faces]")
            t1 = time.time()
            print(f"  [{round(t1-t0, 3)} secs]")

        # Save mesh as a PLY file
        print(" -Saving PLY file...")
        t0 = time.time()

        # Prepare the correct file name and path
        if save_path is None:
            if outfile_basename != "":
                outfile_path = f"{outfile_basename}_label{label}.ply"
                outfile_path = os.path.join(os.path.dirname(args.path), save_subfolder, outfile_path)
            else:
                outfile_path = os.path.join(save_subfolder,f"{os.path.splitext(args.path)[0]}_label{label}.ply")
        else:
            if outfile_basename != "":
                outfile_path = f"{outfile_basename}_label{label}.ply"
                outfile_path = os.path.join(save_path, save_subfolder, outfile_path)
            else:
                outfile_path = os.path.splitext(args.path)[0]
                outfile_path = f"{os.path.basename(outfile_path)}_label{label}.ply"
                outfile_path = os.path.join(save_path, save_subfolder, outfile_path)

        os.makedirs(os.path.dirname(outfile_path), exist_ok=True)
        # Export a PLY
        writePLYfile(vtkPoly, savepath = outfile_path)

        t1 = time.time()
        print(f"  [{round(t1-t0, 3)} secs]")
        print(f"{25*'-'}")
        return 1
    else:
        return 0


def label2mesh_mp(iterator_element):
    """wraper around label2mesh for multiprocessing and non-blocking error handling"""
    segmentation, label, min_vol, save_path, outfile_basename, save_subfolder = iterator_element
    try:
        result = label2mesh(segmentation, label, min_vol, save_path, outfile_basename, save_subfolder)
    except:
        print(f'Runtime exception while processing label: {label}.')
        print(traceback.format_exc())
        result = 0
    return result

def labels2meshes(segmentation, labels_list, min_vol=0, save_path=None, outfile_basename="", save_subfolder=""):
    if _multiprocessing > 1:
        print(f"Using {_multiprocessing} cores out of {num_cpus}\n{50*'='}")
        iterator = [(segmentation,
                    label,
                    min_vol,
                    save_path,
                    outfile_basename,
                    save_subfolder) for label in labels_list]

        with futures.ProcessPoolExecutor(max_workers=_multiprocessing) as executor:
            results = [result for result in executor.map(label2mesh_mp, iterator)]
        return sum(results)
    else:
        results = []
        results = [label2mesh_mp((segmentation, label, min_vol, save_path, outfile_basename, save_subfolder)) for label in labels_list]
        return sum(results)


def args_parser():
    parser = argparse.ArgumentParser(description='Simple utility for generating ply mesh(es)'
                                                 'from label(s) in a h5 segmentation as generated by PlantSeg')
    # Mandatory arguments
    parser.add_argument('--path', type=str, help='Path to the segmentation file (only h5).',
                        required=True)
    parser.add_argument('--dataset', type=str, help='Name of the h5 dataset to retrieve the labels from (use h5ls to see which exist)',
                        default="segmentation", required=True)
    # Optional arguments
    # Retrieve specific labels
    parser.add_argument('--labels', type=int, help='Labels id to extract (example: --labels 10 25 100).',
                        required=False, nargs='+')
    # Retrieve all labels
    parser.add_argument('--all', help='Retrieve all labels', action='store_true')
    # Filter labels based on size
    parser.add_argument('--min-volume', type=int, help='Minimal volume (voxels) of labels for extraction.', required=False, default = 0)
    # Specify output path & base name
    parser.add_argument('--out-path', type=str, help='Path to alternative save directory', default=None, required=False)
    parser.add_argument('--out-name', type=str, help='Use this as base name for output file(s).', default="", required=False)
    # Mesh post-processing
    parser.add_argument('--reduction', type=float, help='If reduction > 0 a decimation filter is applied.' ' MaxValue: 1.0 (100%reduction).', default=-.0, required=False)
    parser.add_argument('--smoothing', help='To apply a Laplacian smoothing filter.', action='store_true')
    parser.add_argument('--step-size', help='Step size for the marching cube algorithm, larger steps yield a coarser but faster result. Default 2 (voxel).',
                        default=2, required=False)
    # Multiprocessing
    parser.add_argument('--multiprocessing', help='Define the number of cores to use for parallel processing.', required=False, default=1, type=int)
    # Batch Modes:
    # Specific files & labels (from a TSV file)
    parser.add_argument('--batch', type=str, help='Batch process several h5 files. Pass path to a tab-delimited file for time points and labels.', default="", required=False)
    # All labels in all files
    parser.add_argument('--batch-all', help='Retrieve all labels in all files.'
                                            'Script will attempt to process all time points based on the file passed in --path', action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    # Pars and check inputs
    args = args_parser()

    assert os.path.isfile(args.path), "Path is not a file"

    if args.out_path is not None:
        assert os.path.isdir(args.out_path), "Save path is not a directory"

    _process_all_labels = args.all
    _process_all_labels_batch = args.batch_all
    _batch = True if args.batch !="" else False
    _label = ""
    _labels_tsv = args.batch
    _dataset = args.dataset
    _out_name = args.out_name
    _reduction = args.reduction
    _smoothing = args.smoothing
    _step_size = args.step_size
    _min_vol = args.min_volume
    _multiprocessing = args.multiprocessing

    num_cpus = psutil.cpu_count(logical=False)
    _multiprocessing = _multiprocessing if _multiprocessing < num_cpus else num_cpus

    total_processed = 0

    t0 = time.time()
    print(f"{50*'='}\nseg2mesh v{_version}")
    if _process_all_labels_batch:
        _pattern_found = re.match("(^.*[tT]\d{3,})\d{2}(.*)", args.path)
        if _pattern_found:
            _regex_frgt1 = _pattern_found.group(1)
            _regex_frgt2 = _pattern_found.group(2)
        else:
            print(" -Error: file name does not contain recognisable time pattern (tXXXXX)")
        p = Path(args.path)
        # FIXME ?? problem when passing a complete path to file  -no problem when passing file name directly
        for h5_file in sorted(list(p.parents[0].rglob('*.h5'))):
            _pattern_found = re.match(f"{_regex_frgt1}(.*){_regex_frgt2}", h5_file.name)
            if _pattern_found:
                time_point = _pattern_found.group(1)
                _simple_name_batch = f"{_out_name}_t{time_point}"
                segmentation = get_segmentation(h5_file.as_posix(), dataset = _dataset)
                _labels = get_all_labels(segmentation, dataset=_dataset)
                total_processed = labels2meshes(segmentation,
                                                _labels,
                                                min_vol=_min_vol,
                                                save_path=args.out_path,
                                                outfile_basename=_out_name,
                                                save_subfolder=f"t{time_point}")
    elif _batch:
        _pattern_found = re.match("(^.*[tT]\d{3,})\d{2}(.*)", args.path)
        if _pattern_found:
            _regex_frgt1 = _pattern_found.group(1)
            _regex_frgt2 = _pattern_found.group(2)
            with open(_labels_tsv) as tsv:
                next(tsv) # skip headings
                for time_point, labels in csv.reader(tsv, delimiter="\t"):
                    _inpath = f"{_regex_frgt1}{time_point}{_regex_frgt2}"
                    _simple_name_batch = f"{_out_name}_t{time_point}"
                    _labels = labels.split()
                    segmentation = get_segmentation(_inpath.as_posix(), dataset = _dataset)
                    total_processed =labels2meshes(segmentation, _labels, min_vol = _min_vol, save_path = args.out_path, outfile_basename = _out_name, save_subfolder=f"t{time_point}")
        else:
            "Input file is not of the correct format."
    else:
        segmentation = get_segmentation(args.path, dataset = _dataset)
        if _process_all_labels :
            _labels = get_all_labels(segmentation, dataset=_dataset)
        else:
            _labels = args.labels
        total_processed = labels2meshes(segmentation, _labels, min_vol = _min_vol, save_path = args.out_path, outfile_basename = _out_name)

    t1 = time.time()
    print(f"{50*'='} \nFinished!")
    print(f"{total_processed} objects in {datetime.timedelta(seconds=(t1-t0))} [HH:mm:ss]")