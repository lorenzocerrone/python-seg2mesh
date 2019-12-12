import numpy as np
from skimage import measure
import h5py
import os
import csv
import re
import plyfile
import argparse
from pathlib import Path

# TODO  This is a very slow operation, why?- speed up?
def clean_object(obj):
    """If object has more than one connected component returns only biggest components"""
    print("- Cleaning small detached objects")
    # relabel connected components
    obj_relabeled = measure.label(obj, background=0)

    # find biggest object id
    ids, counts = np.unique(obj_relabeled, return_counts=True)
    ids, counts = ids[1:], counts[1:]
    largest_id = ids[np.argmax(counts)]

    # select single object
    obj_clean = obj_relabeled == largest_id
    return obj_clean


def compute_obj(segmentation, label):
    """Extract a mask where the label is"""
    print(f"- Extracting object: {label}")
    obj = segmentation == label
    return obj


def single_obj_mesh(segmentation, label, step_size):
    """Compute a mesh from a single object"""
    obj = compute_obj(segmentation, label)

    if not np.any(obj):
        # If no index match nothing to do
        print(f"- Label: {label} Not found")
        return None, None

    obj = clean_object(obj)
    obj = obj.astype(float)

    # Extract vertex and faces
    vertx, faces, _, _ = measure.marching_cubes_lewiner(obj, step_size=step_size)
    return vertx, faces


def multi_obj_mesh(segmentation, labels, step_size):
    """Concatenate multiple objects in a single mesh"""
    vertx, faces = [], []
    faces_max = 0
    for label in labels:
        obj = compute_obj(segmentation, label)
        if np.any(obj):
            obj = clean_object(obj)
            obj = obj.astype(float)

            # Extract vertex and faces
            _vertx, _faces, _, _ = measure.marching_cubes_lewiner(obj, step_size=step_size)

            # Add max to ensure unique faces
            _faces += faces_max

            faces_max = _faces.max() + 1
            vertx.append(_vertx)
            faces.append(_faces)
        else:
            print(f"- Label: {label} Not found")

    vertx = np.concatenate(vertx, axis=0)
    faces = np.concatenate(faces, axis=0)

    return vertx, faces

def get_all_labels(path, dataset):
    print("- Getting all labels")
    print(f"- Loading segmentation from :{path}")
    with h5py.File(path, "r") as f:
        segmentation = f[dataset][...]
        all_labels= np.unique(segmentation)
        print(f"- Found {len(all_labels)} labels")
        return all_labels

def label2mesh(path, label, multi_file=True, save_path=None, center_origin=False, dataset="label", step_size=1, outputbasename="", save_subfolder=""):
    print(f"- Loading segmentation from :{path}")
    with h5py.File(path, "r") as f:
        segmentation = f[dataset][...]

    print(f"- Object extraction and computing marching cubes")
    # if needed more than single label a different criteria can be used
    if multi_file:
        vertx, faces = single_obj_mesh(segmentation, label, step_size)
    else:
        vertx, faces = multi_obj_mesh(segmentation, label, step_size)

    # If no index match nothing to do
    if vertx is None:
        return 0

    if center_origin and multi_file:
        mean_zxy = np.mean(vertx, axis=0)
    else:
        mean_zxy = np.array([0, 0, 0])

    print('- Creating ply file')
    # Create vertex attributes
    vertex_attributes = []
    for i in range(vertx.shape[0]):
        vertex_attributes.append((vertx[i, 0] - mean_zxy[0],
                                  vertx[i, 1] - mean_zxy[1],
                                  vertx[i, 2] - mean_zxy[2]))

    vertex_attributes = np.array(vertex_attributes, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_attributes = plyfile.PlyElement.describe(vertex_attributes, 'vertex')

    # Create face attributes
    faces_attributes = []
    for i in range(faces.shape[0]):
        faces_attributes.append(([faces[i, 0], faces[i, 1], faces[i, 2]], 0, 255, 0))

    faces_attributes = np.array(faces_attributes, dtype=[('vertex_indices', 'i4', (3,)),
                                                         ('red', 'u1'),
                                                         ('green', 'u1'),
                                                         ('blue', 'u1')])
    faces_attributes = plyfile.PlyElement.describe(faces_attributes, 'face')

    # if no path is specified the ply is create next to the original file
    if not multi_file:
        label = "".join(map(lambda x: f"_{x}", label))

    if save_path is None:
        if outputbasename != "":
            new_file = f"{outputbasename}_label{label}.ply"
            new_file = os.path.join(os.path.dirname(path), save_subfolder, new_file)
        else:
            new_file = os.path.join(save_subfolder,f"{os.path.splitext(path)[0]}_label{label}.ply")
    else:
        if outputbasename != "":
            new_file = f"{outputbasename}_label{label}.ply"
            new_file = os.path.join(save_path, save_subfolder, new_file)
        else:
            new_file = os.path.splitext(path)[0]
            new_file = f"{os.path.basename(new_file)}_label{label}.ply"
            new_file = os.path.join(save_path, save_subfolder, new_file)

    print(f"  -> Saving file at: {new_file}")
    os.makedirs(os.path.dirname(new_file), exist_ok=True)
    plyfile.PlyData((vertex_attributes, faces_attributes)).write(new_file)
    return new_file


def _parser():
    parser = argparse.ArgumentParser(description='Simple utility for extracting a single '
                                                 'label from a h5 segmentation into a ply mesh')
    parser.add_argument('--path', type=str, help='Path to the segmentation file (only h5).',
                        required=True)
    parser.add_argument('--dataset', type=str, help='Name of the h5 dataset to retrieve the labels from (use h5ls to see which exist)',
                        default="label", required=True)
    parser.add_argument('--labels', type=int, help='Labels id to extract (example: --labels 10 25 100).',
                        required=False, nargs='+')
    parser.add_argument('--single-file', help='All meshes are saved in the same file',
                        action='store_true')
    parser.add_argument('--save-path', type=str, help='Path to alternative save directory',
                        default=None, required=False)
    parser.add_argument('--center-origin', type=str, help='Shift the mesh to the axis origin',
                        default="False", required=False)
    parser.add_argument('--step-size', type=int, help='Marching cube step size (int). The higher the coarser the output'
                                                      ' mesh. Default: 1 (full resolution)',
                        default=1, required=False)
    # Simple Name Mode
    parser.add_argument('--simple-name', type=str, help='Use this as base name for output file(s).', default="", required=False)
    # Batch Mode
    parser.add_argument('--batch', type=str, help='Batch process several h5 files. Pass path to a tab-delimited file for time points and labels. Forces --multi-file TRUE.', default="", required=False)
    # Retrieve all labels
    parser.add_argument('--all', help='Retrieve all labels', action='store_true')
    # Retrieve all labels in all files
    parser.add_argument('--batch-all', help='Retrieve all labels in all files.'
                                            'Script will attempt to process all time points based on the file passed in --path', action='store_true')
    # Post processing
    parser.add_argument('--reduction', type=float, help='If reduction > 0 a decimation filter is applied.'
                                                        ' MaxValue: 1.0 (100%reduction).',
                        default=-.0, required=False)

    parser.add_argument('--iterations', type=int, help='If iteration > 0 a Laplacian smoothing filter is applied.',
                        default=0, required=False)
    parser.add_argument('--relaxation', type=float, help='The smaller the better accuracy but slower convergence.'
                                                         ' Default: 0.1',
                        default=0.1, required=False)
    parser.add_argument('--edge-smoothing', help='Apply edge smoothing. Default False,'
                                                          ' seems to help after very intensive decimation',
                        action='store_true', required=False)


    return parser.parse_args()


if __name__ == "__main__":
    # Pars and check inputs
    args = _parser()
    assert os.path.isfile(args.path), "Path is not a file"

    if args.save_path is not None:
        assert os.path.isdir(args.save_path), "Save path is not a directory"

    _center_origin = True if args.center_origin == "True" else False
    _process_all_labels  = args.all
    _process_all_labels_batch  = args.batch_all
    _single_file = args.single_file
    _batch = True if args.batch !="" else False
    if _batch: _single_file = False
    _label = ""
    _lables_tsv = args.batch
    _dataset = args.dataset
    _step_size = args.step_size
    _simple_name = args.simple_name

    out_path = []

    if not _single_file:
        if _process_all_labels_batch:
            _pattern_found = re.match("(^.*[tT]\d{3,})\d{2}(.*)", args.path)
            if _pattern_found:
                _regex_frgt1 = _pattern_found.group(1)
                _regex_frgt2 = _pattern_found.group(2)
            else:
                print("- Error: file name does not contain recognisable time pattern (tXXXXX)")
            # list all files in directory containing the file passed to args.path
            p = Path(args.path)
            for h5_file in sorted(list(p.parents[0].rglob('*.h5'))):
                _pattern_found = re.match(f"{_regex_frgt1}(.*){_regex_frgt2}", h5_file.name)
                if _pattern_found:
                    time_point = _pattern_found.group(1)
                    print(f"{50*'='} \nProcessing file: {h5_file.name}")
                    _simple_name_batch = f"{_simple_name}_t{time_point}"
                    _labels = get_all_labels(h5_file.as_posix(), dataset=_dataset)
                    # Run main script over all labels for multiple files
                    for label in _labels:
                        print(f"Extracting Label: {int(label)}")
                        _path = label2mesh(h5_file.as_posix(),
                                            int(label),
                                            multi_file=True,
                                            save_path=args.save_path,
                                            center_origin=_center_origin,
                                            dataset=_dataset,
                                            step_size=_step_size,
                                            outputbasename=_simple_name_batch,
                                            save_subfolder=f"t{time_point}"
                                            )
                        out_path.append(_path)
        elif _batch:
            _pattern_found = re.match("(^.*[tT]\d{3,})\d{2}(.*)", args.path)
            if _pattern_found:
                _regex_frgt1 = _pattern_found.group(1)
                _regex_frgt2 = _pattern_found.group(2)
                with open(_lables_tsv) as tsv:
                    next(tsv) # skip headings
                    for time_point, labels in csv.reader(tsv, delimiter="\t"):
                        _inpath = f"{_regex_frgt1}{time_point}{_regex_frgt2}"
                        print(f"{50*'='} \nProcessing file: {_inpath}")
                        _simple_name_batch = f"{_simple_name}_t{time_point}"
                        if _process_all_labels :
                            _labels = get_all_labels(args.path, dataset=_dataset)
                        else:
                            _labels = labels.split()
                        # Run main script over all labels for multiple files
                        for label in _labels:
                            print(f"Extracting Label: {int(label)}")
                            _path = label2mesh(_inpath,
                                               int(label),
                                               multi_file=True,
                                               save_path=args.save_path,
                                               center_origin=_center_origin,
                                               dataset=_dataset,
                                               step_size=_step_size,
                                               outputbasename=_simple_name_batch,
                                               save_subfolder=f"t{time_point}")
                            out_path.append(_path)
            else:
                "Input file is not of the correct format."
        else:
            # Run main script over all labels for multiple files
            out_path = []
            if _process_all_labels :
                _labels_list = get_all_labels(args.path, dataset=_dataset)
            else:
                _labels_list = args.labels
            for _label in _labels_list:
                print(f"{50*'='} \nExtracting Label: {_label}")
                _path = label2mesh(args.path,
                                   _label,
                                   multi_file=True,
                                   save_path=args.save_path,
                                   center_origin=_center_origin,
                                   dataset=_dataset,
                                   step_size=_step_size,
                                   outputbasename=_simple_name)
                out_path.append(_path)
    else:
        _path = label2mesh(args.path,
                           args.labels,
                           multi_file=False,
                           save_path=args.save_path,
                           center_origin=_center_origin,
                           dataset=_dataset,
                           step_size=_step_size,
                           outputbasename=_simple_name)
        out_path = [_path]

    print(f"{50*'='} \nPost processing")
    from plyfilters import decimation, smooth

    for path in out_path:
        assert args.reduction < 1, "Reduce factor cannot be larger than 1 (more than 100% reduction)"
        if args.reduction > 0:
            print("- Applying decimation")
            out = decimation(path, args.reduction, args.save_path)
        else:
            out = path

        if args.iterations > 0:
            print("- Applying Laplacian smoothing")
            _edgemoothing = args.edge_smoothing
            smooth(out, args.iterations, args.relaxation, _edgemoothing, args.save_path)
    print(f"{50*'='} \nFinished")
