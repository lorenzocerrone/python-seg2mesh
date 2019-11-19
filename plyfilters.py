import os
import argparse
from vtk import vtkPolyData, vtkPLYReader, vtkPLYWriter, vtkDecimatePro, vtkSmoothPolyDataFilter, vtkPolyDataNormals



def decimation(path, reduction=0.25, savepath=None):
    """ Edit a mesh file (.ply format) reducing the density of the mesh using decimation"""
    # source https://lorensen.github.io/VTKExamples/site/Cxx/Meshes/Decimation/

    # read Ply file
    reader = vtkPLYReader()
    reader.SetFileName(path)
    reader.Update()

    # create Poly data
    inputPoly = vtkPolyData()
    inputPoly.ShallowCopy(reader.GetOutput())

    # decimate and copy data
    decimate = vtkDecimatePro()
    decimate.SetInputData(inputPoly)
    decimate.SetTargetReduction(reduction)  # (float) set 0 for no reduction and 1 for 100% reduction
    decimate.Update()

    decimatedPoly = vtkPolyData()
    decimatedPoly.ShallowCopy(decimate.GetOutput())

    # write results on output file
    if savepath is None:
        outfile_path = os.path.splitext(path)[0]
        outfile_path = f"{outfile_path}_decimation_{int(reduction * 100)}.ply"
    else:
        outfile_path = os.path.splitext(path)[0]
        outfile_path = f"{os.path.basename(outfile_path)}_decimation_{int(reduction * 100)}.ply"
        outfile_path = os.path.join(savepath, outfile_path)

    writer = vtkPLYWriter()
    writer.SetInputData(decimatedPoly)
    writer.SetFileName(outfile_path)
    writer.Write()
    print(f" - saving file at: {outfile_path}")
    return outfile_path


def smooth(path, iterations=100, relaxation=0.1, edgesmoothing=True, savepath=None):
    """ Edit a mesh file (ply format) applying iterative Laplacian smoothing """
    # source https://vtk.org/Wiki/VTK/Examples/Cxx/PolyData/SmoothPolyDataFilter

    # read Ply file
    reader = vtkPLYReader()
    reader.SetFileName(path)
    reader.Update()

    # create Poly data
    inputPoly = vtkPolyData()
    inputPoly.ShallowCopy(reader.GetOutput())

    # Smooth mesh with Laplacian Smoothing
    smooth = vtkSmoothPolyDataFilter()
    smooth.SetInputData(inputPoly)
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

    # write results on output file
    if savepath is None:
        outfile_path = os.path.splitext(path)[0]
        outfile_path = f"{outfile_path}_smooth_{iterations}.ply"
    else:
        outfile_path = os.path.splitext(path)[0]
        outfile_path = f"{os.path.basename(outfile_path)}_smooth_{iterations}.ply"
        outfile_path = os.path.join(savepath, outfile_path)

    writer = vtkPLYWriter()
    writer.SetInputData(normalPoly)
    writer.SetFileName(outfile_path)
    writer.Write()
    print(f" - saving file at: {outfile_path}")
    return outfile_path


def _parser():
    parser = argparse.ArgumentParser(description='Smoothing and Decimation utility for ply files')
    parser.add_argument('--path', type=str, help='Path to the mesh file (only ply)',
                        required=True)

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

    parser.add_argument('--save-path', type=str, help='Path to alternative save directory',
                        default=None, required=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = _parser()
    assert args.reduction < 1, "mesh can not be reduce factor cannot be larger than 1 (more than 100% reduction)"
    if args.reduction > 0:
        print("- Applying decimation")
        out = decimation(args.path, args.reduction, args.save_path)
    else:
        out = args.path

    if args.iterations > 0:
        print("- Applying Laplacian smoothing")
        _edgemoothing = True if args.edge_smoothing == "True" else False
        smooth(out, args.iterations, args.relaxation, _edgemoothing, args.save_path)
