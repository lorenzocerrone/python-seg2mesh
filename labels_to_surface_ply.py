import numpy as np
from skimage import measure
import h5py
import os
import plyfile
import argparse
from numba import jit


_dataset = "segmentation"


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


def multi_obj_mesh(segmentation, labels):
    """Concatenate multiple objects in a single mesh"""
    vertx, faces, colors = [], [], []
    faces_max = 0
    for label in labels:
        obj = compute_obj(segmentation, label)
        if np.any(obj):
            obj = clean_object(obj)
            obj = obj.astype(float)

            # Extract vertex and faces
            _vertx, _faces, _, _ = measure.marching_cubes_lewiner(obj)

            # Add max to ensure unique faces
            _faces += faces_max

            faces_max = _faces.max() + 1

            _colors = np.random.randint(255, size=3)[None, :] * np.ones((_faces.shape[0], 3))

            vertx.append(_vertx)
            faces.append(_faces)
            colors.append(_colors)
        else:
            print(f"- Label: {label} Not found")

    vertx = np.concatenate(vertx, axis=0)
    faces = np.concatenate(faces, axis=0)
    colors = np.concatenate(colors, axis=0)

    return vertx, faces, colors


@jit
def extract_adjacent(segmentation, background_id):
    zshape, xshape, yshape = segmentation.shape
    labels_list = []
    for z in range(zshape - 1):
        for x in range(xshape - 1):
            for y in range(yshape - 1):
                _id = segmentation[z, x, y]
                if _id == background_id:
                    z_plus = segmentation[z + 1, x, y]
                    x_plus = segmentation[z, x + 1, y]
                    y_plus = segmentation[z, x, y + 1]

                    if z_plus != _id:
                        labels_list.append(z_plus)
                    if x_plus != _id:
                        labels_list.append(x_plus)
                    if y_plus != _id:
                        labels_list.append(y_plus)

    return labels_list


def surface_list(segmentation):
    ids, counts = np.unique(segmentation, return_counts=True)
    background_id = ids[np.argmax(counts)]

    labels_list = extract_adjacent(segmentation, background_id=background_id)
    labels_list = np.unique(np.array(labels_list))
    return labels_list


def _parser():
    parser = argparse.ArgumentParser(description='Simple utility for extracting a single '
                                                 'label from a h5 segmentation into a ply mesh')
    parser.add_argument('--path', type=str, help='Path to the segmentation file (only h5)',
                        required=True)
    parser.add_argument('--save-path', type=str, help='Path to alternative save directory',
                        default=None, required=False)
    return parser.parse_args()


def surface2mesh(path, save_path=None):
    print(f"- Loading segmentation from :{path}")
    with h5py.File(path, "r") as f:
        segmentation = f[_dataset][...]

    print(f"- Extracting surface ids")
    label = surface_list(segmentation)

    print(f"- Object extraction and computing marching cubes")
    # if needed more than single label a different criteria can be used
    vertex, faces, colors = multi_obj_mesh(segmentation, label)

    # If no index match nothing to do
    if vertex is None:
        return 0

    print('- Creating ply file')
    # Create vertex attributes
    vertex_attributes = []
    for i in range(vertex.shape[0]):
        vertex_attributes.append((vertex[i, 0],
                                  vertex[i, 1],
                                  vertex[i, 2]))

    vertex_attributes = np.array(vertex_attributes, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_attributes = plyfile.PlyElement.describe(vertex_attributes, 'vertex')

    # Create face attributes
    faces_attributes = []
    for i in range(faces.shape[0]):
        faces_attributes.append(([faces[i, 0], faces[i, 1], faces[i, 2]], colors[i, 0], colors[i, 1], colors[i, 2]))

    faces_attributes = np.array(faces_attributes, dtype=[('vertex_indices', 'i4', (3,)),
                                                         ('red', 'u1'),
                                                         ('green', 'u1'),
                                                         ('blue', 'u1')])
    faces_attributes = plyfile.PlyElement.describe(faces_attributes, 'face')

    # if no path is specified the ply is create next to the original file

    if save_path is None:
        new_file = f"{os.path.splitext(path)[0]}_surface.ply"
    else:
        new_file = os.path.splitext(path)[0]
        new_file = f"{os.path.basename(new_file)}_surface.ply"
        new_file = os.path.join(save_path, new_file)

    print(f"- saving file at: {new_file}")
    plyfile.PlyData((vertex_attributes, faces_attributes)).write(new_file)


if __name__ == "__main__":
    # Pars and check inputs
    args = _parser()
    assert os.path.isfile(args.path), "Path is not a file"

    if args.save_path is not None:
        assert os.path.isdir(args.save_path), "Save path is not a directory"

    surface2mesh(args.path, save_path=args.save_path)
