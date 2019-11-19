import numpy as np
from skimage import measure
import h5py
import os
import plyfile
import argparse


def clean_object(obj):
    """If object has more than one connected component returns only biggest components"""
    print(" - Cleaning small detached objects")
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


def label2mesh(path, label, multi_file=True, save_path=None, center_origin=False, dataset="label", step_size=1):
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
        new_file = f"{os.path.splitext(path)[0]}_label{label}.ply"
    else:
        new_file = os.path.splitext(path)[0]
        new_file = f"{os.path.basename(new_file)}_label{label}.ply"
        new_file = os.path.join(save_path, new_file)

    print(f"- saving file at: {new_file}")
    plyfile.PlyData((vertex_attributes, faces_attributes)).write(new_file)
    return new_file


def _parser():
    parser = argparse.ArgumentParser(description='Simple utility for extracting a single '
                                                 'label from a h5 segmentation into a ply mesh')
    parser.add_argument('--path', type=str, help='Path to the segmentation file (only h5)',
                        required=True)
    parser.add_argument('--labels', type=int, help='Labels id to extract (example: --labels 10 25 100)',
                        required=True, nargs='+')
    parser.add_argument('--multi-file', type=str, help='If "True" all meshes are saved in a different file',
                        default="False", required=False)
    parser.add_argument('--save-path', type=str, help='Path to alternative save directory',
                        default=None, required=False)
    parser.add_argument('--center-origin', type=str, help='Shift the mesh to the axis origin',
                        default="False", required=False)
    parser.add_argument('--dataset', type=str, help='Name of the h5 dataset to retrieve the labels from',
                        default="label", required=False)
    parser.add_argument('--step-size', type=int, help='Marching cube step size (int). The higher the coarser the output'
                                                      ' mesh. Default 1 (full resolution)',
                        default=1, required=False)

    # Post processing
    parser.add_argument('--reduction', type=float, help='If reduction > 0 a decimation filter is applied.'
                                                        ' MaxValue 1.0 (100%reduction).',
                        default=-.0, required=False)

    parser.add_argument('--iterations', type=int, help='If iteration > 0 a Laplacian smoothing filter is applied.',
                        default=0, required=False)
    parser.add_argument('--relaxation', type=float, help='The smaller the better accuracy but slower convergence.'
                                                         ' Default 0.1',
                        default=0.1, required=False)
    parser.add_argument('--edge-smoothing', type=str, help='Apply edge smoothing. Default False,'
                                                          ' seems to help after very intensive decimation',
                        default="False", required=False)
    return parser.parse_args()


if __name__ == "__main__":
    # Pars and check inputs
    args = _parser()
    assert os.path.isfile(args.path), "Path is not a file"

    if args.save_path is not None:
        assert os.path.isdir(args.save_path), "Save path is not a directory"

    _center_origin = True if args.center_origin == "True" else False
    _multi_file = True if args.multi_file == "True" else False
    _label = ""
    _dataset = args.dataset
    _step_size = args.step_size

    if _multi_file:
        # Run main script over all labels for multiple files
        out_path = []
        for _label in args.labels:
            print(f"{50*'='} \nExtracting Label: {_label}")
            _path = label2mesh(args.path,
                               _label,
                               multi_file=True,
                               save_path=args.save_path,
                               center_origin=_center_origin,
                               dataset=_dataset,
                               step_size=_step_size)
            out_path.append(_path)
    else:
        _path = label2mesh(args.path,
                           args.labels,
                           multi_file=False,
                           save_path=args.save_path,
                           center_origin=_center_origin,
                           dataset=_dataset,
                           step_size=_step_size)
        out_path = [_path]

    print('Post processing')
    """ Huge hack but vtk python wrapper creates a lot of import problems"""
    from subprocess import call

    for path in out_path:
        if args.save_path is None:
            call(["python", "./plyfilters.py",
                  "--path", path,
                  "--reduction", str(args.reduction),
                  "--iterations", str(args.iterations),
                  "--relaxation", str(args.relaxation),
                  "--edge-smoothing", str(args.edge_smoothing)])
        else:
            call(["python", "./plyfilters.py",
                  "--path", path,
                  "--reduction", str(args.reduction),
                  "--iterations", str(args.iterations),
                  "--relaxation", str(args.relaxation),
                  "--edge-smoothing", str(args.edge_smoothing),
                  "--save-path", args.save_path])
