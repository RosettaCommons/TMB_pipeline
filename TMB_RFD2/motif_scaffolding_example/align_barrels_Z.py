#! /software/containers/users/sagardip/TMBproject.sif

"""
Computes principal axes from a PDB file.

Produces also a .pml script for a nice rendering with PyMOL.
"""

import sys
import os.path
import numpy
from scipy.linalg import expm
from Bio.PDB import *


# scale factor to enhance the length of axis in Pymol
scale_factor = 20

def scale_func(x):
    if x <= 5.0:
        y = x
    else:
        y = 1/x**3
    return y

def mod_coords(coords):
    factor = []
    norm = numpy.linalg.norm(coords, axis=1)
    for r in norm:
        factor.append(scale_func(r))

    new_coords = (coords.T*numpy.array(factor)).T
    return new_coords

def make_rotmat(vec):
    # Calculate the cross product and dot product of the input and target vectors
    cross_product = numpy.cross(vec, numpy.array([0,0,1]))
    dot_product = numpy.dot(vec, numpy.array([0,0,1]))

    # Calculate the skew-symmetric matrix
    skew_matrix = numpy.array([[0, -cross_product[2], cross_product[1]],
                            [cross_product[2], 0, -cross_product[0]],
                            [-cross_product[1], cross_product[0], 0]])

    # Calculate the rotation matrix
    rotation_matrix = numpy.identity(3) + skew_matrix + numpy.matmul(skew_matrix, skew_matrix) / (1 + dot_product)


#    theta = numpy.arccos(vec[2])
#    return expm(numpy.cross(numpy.eye(3), theta*vec))
    return rotation_matrix

def invert_pdb(coord):
    rotmat = numpy.array([[-1,0,0],[0,1,0],[0,0,-1]])
    coord = numpy.dot(coord, rotmat.T)
    return coord



def read_pdb_coord(pdb_name):
    """
    Reads atomic coordinates of C-alpha atoms from a .pdb file.

    Parameters
    ----------
    pdb_name : str
        Name of pdb file.

    Returns
    -------
    array of atomic coordinates
        [[x1 y1 z1]
         [x2 y2 z2]
         [.. .. ..]
         [xn yn zn]]
    """
    p = PDBParser()
    st = p.get_structure('pdb', pdb_name)

    coordinates = []
    ca = []
    for model in st:
        for chain in model:
            for residue in chain:
                if residue.has_id('CA'):
                    atom = residue['CA']
                    ca.append(atom.coord)
                for atom in residue:
                    coordinates.append(atom.coord)

    return numpy.array(ca), numpy.array(coordinates), st


def dump_pdb(pdb, name):
    io = PDBIO()
    io.set_structure(pdb)
    io.save(name)


# start program
if __name__ == '__main__':
    #--------------------------------------------------------------------------
    # center axes to the geometric center of the molecule
    # and rescale them by order of eigen values
    #--------------------------------------------------------------------------
    # the large vector is the first principal axis
    point1 = 1 * scale_factor * numpy.array([1,0,0])
    # the medium vector is the second principal axis
    point2 = 1 * scale_factor * numpy.array([0,1,0])
    # the small vector is the third principal axis
    point3 = 5 * scale_factor * numpy.array([0,0,1])

    #--------------------------------------------------------------------------
    # create .pml script for a nice rendering in Pymol
    #--------------------------------------------------------------------------
    pymol_name = "axes.pml"
    with open(pymol_name, "w") as pymol_file:
        pymol_file.write(
            """
            from pymol.cgo import *
            axis1=  [ BEGIN, LINES, COLOR, 1.0, 0.0, 0.0, \
            VERTEX, %8.3f, %8.3f, %8.3f, VERTEX, %8.3f, %8.3f, %8.3f, END ]
            axis2=  [ BEGIN, LINES, COLOR, 0.0, 1.0, 0.0, \
            VERTEX, %8.3f, %8.3f, %8.3f, VERTEX, %8.3f, %8.3f, %8.3f, END ]
            axis3=  [ BEGIN, LINES, COLOR, 0.0, 0.0, 1.0, \
            VERTEX, %8.3f, %8.3f, %8.3f, VERTEX, %8.3f, %8.3f, %8.3f, END ]
            cmd.load_cgo(axis1, 'axis1')
            cmd.load_cgo(axis2, 'axis2')
            cmd.load_cgo(axis3, 'axis3')
            cmd.set('cgo_line_width', 4)
            """ %( \
                    0.0, 0.0, 0.0, point1[0], point1[1], point1[2], \
                    0.0, 0.0, 0.0, point2[0], point2[1], point2[2], \
                    0.0, 0.0, 0.0, point3[0], point3[1], point3[2]))



    for pdb_name in sys.argv[1:]:
    #--------------------------------------------------------------------------
    # compute principal axes
    #--------------------------------------------------------------------------
    # read pdb
        coord_ca, allcoord, pdbst = read_pdb_coord(pdb_name)

    # compute geometric center
        center = numpy.mean(coord_ca, 0)

    # center with geometric center
        coord = mod_coords(coord_ca - center)

    # compute principal axis matrix
        inertia = numpy.dot(coord.transpose(), coord)
        e_values, e_vectors = numpy.linalg.eig(inertia)
    #--------------------------------------------------------------------------
    # order eigen values (and eigen vectors)
    #
    # axis1 is the principal axis with the biggest eigen value (eval1)
    # axis2 is the principal axis with the second biggest eigen value (eval2)
    # axis3 is the principal axis with the smallest eigen value (eval3)
    #--------------------------------------------------------------------------
        order = numpy.argsort(e_values)
        eval3, eval2, eval1 = e_values[order]
        axis3, axis2, axis1 = e_vectors[:, order].transpose()
    
#        v = numpy.cross(numpy.array([0,0,1]), axis3)
#        norm_v = v/numpy.linalg.norm(v)

#        print(norm_v)

        rotmat = make_rotmat(axis3/numpy.linalg.norm(axis3))

        moved_coord = numpy.dot(allcoord-center, rotmat.T)

        if moved_coord[0][2] > 0.0:
            moved_coord = invert_pdb(moved_coord)

        count = 0
        for model in pdbst:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        atom.coord = list(moved_coord[count])
                        count += 1

        outname = "./alnZ_" + pdb_name.split("/")[-1]
        dump_pdb(pdbst, outname)


    print("\nYou can view alignment to Z-axis by loading the axes.pml file as: pymol axes.pml")
