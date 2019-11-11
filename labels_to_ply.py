import numpy as np
from skimage import measure
import h5py
import os
import plyfile
import argparse

_dataset = "segmentation"


def label2mesh(path, label, save_path=None, center_origin=False):
    print(f"- Loading segmentation from :{path}")
    with h5py.File(path, "r") as f:
        segmentation = f[_dataset][...]

    print(f"- Object extraction and computing marching cubes")
    # if needed more than single label a different criteria can be used
    obj = segmentation == label
    if not np.any(obj):
        print(f"- Label: {label} Not found")
        return 0

    obj = obj.astype(float)

    # Extract vertex and faces

    vertx, faces, _, _ = measure.marching_cubes_lewiner(obj)

    if center_origin:
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
    if save_path is None:
        new_file = f"{os.path.splitext(path)[0]}_label_{label}.ply"
    else:
        new_file = os.path.splitext(path)[0]
        new_file = f"{os.path.basename(new_file)}_label_{label}.ply"
        new_file = os.path.join(save_path, new_file)

    print(f"- saving file at: {new_file}")
    plyfile.PlyData((vertex_attributes, faces_attributes)).write(new_file)


def _parser():
    parser = argparse.ArgumentParser(description='Simple utility for extracting a single '
                                                 'label from a h5 segmentation into a ply mesh')
    parser.add_argument('--path', type=str, help='Path to the segmentation file (only h5)',
                        required=True)
    parser.add_argument('--labels', type=int, help='segments id to extract (example: --labels 10 25 100)',
                        required=True, nargs='+')
    parser.add_argument('--save-path', type=str, help='Path to alternative save directory',
                        default=None, required=False)
    parser.add_argument('--center-origin', type=str, help='Shift the mesh to the axis origin',
                        default="False", required=False)
    return parser.parse_args()


if __name__ == "__main__":
    # Pars and check inputs
    args = _parser()
    assert os.path.isfile(args.path), "Path is not a file"

    if args.save_path is not None:
        assert os.path.isdir(args.save_path), "Save path is not a directory"

    _center_origin = True if args.center_origin == "True" else False

    # Run main script over all labels
    for _label in args.labels:
        print(f"{50*'='} \nExtracting Label: {_label}")
        label2mesh(args.path,
                   _label,
                   save_path=args.save_path,
                   center_origin=_center_origin)
