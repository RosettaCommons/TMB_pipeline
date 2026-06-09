#!/~ENV_DIR/modulus_latest.sif
import sys
import os.path
import json
from pathlib import Path
import numpy
import numpy as np
from scipy.linalg import expm
from Bio.PDB import *

pdbdict = {}
intf_buried_dict = {}
out_file_pdb = "./pdbs_multi_json.jsonl"
out_file_intf_buried = "./intf_buried.jsonl"

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

def invert_pdb(coord1, coord2, coord3):
    rotmat = numpy.array([[-1,0,0],[0,1,0],[0,0,-1]])
    coord1 = numpy.dot(coord1, rotmat.T)
    coord2 = numpy.dot(coord2, rotmat.T)
    coord3 = numpy.dot(coord3, rotmat.T)
    return coord1, coord2, coord3


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
    N_coords = []
    C_coords = []
    ca = []
    for model in st:
        for chain in model:
            for residue in chain:
                if residue.has_id('CA'):
                    atom = residue['CA']
                    ca.append(atom.coord)
                if residue.has_id('N'):
                    atom = residue['N']
                    N_coords.append(atom.coord)
                if residue.has_id('C'):
                    atom = residue['C']
                    C_coords.append(atom.coord)
                for atom in residue:
                    coordinates.append(atom.coord)

    return numpy.array(ca), numpy.array(N_coords), numpy.array(C_coords), numpy.array(coordinates), st



def calculate_cbeta_coordinates(N_coords, C_coords, CA_coords):
    """
    Calculates the idealized C-beta coordinates for a series of residues using
    a rotation-based method to ensure correct tetrahedral geometry.

    Args:
        N_coords (np.ndarray): A numpy array of shape (L, 3) for N coordinates.
        C_coords (np.ndarray): A numpy array of shape (L, 3) for C coordinates.
        CA_coords (np.ndarray): A numpy array of shape (L, 3) for C-alpha coordinates.

    Returns:
        np.ndarray: A numpy array of shape (L, 3) for the idealized C-beta coordinates.
    """
    # Create vectors from C-alpha to N and C
    v_ca_n = N_coords - CA_coords
    v_ca_c = C_coords - CA_coords

    # Normalize the C-alpha to C vector to create the rotation axis
    axis = v_ca_c / np.linalg.norm(v_ca_c, axis=-1, keepdims=True)

    # Set the rotation angle to -120 degrees for a right-handed L-amino acid
    angle = np.deg2rad(-120)

    # Rodrigues' rotation formula components
    # This formula rotates a vector 'v' around an axis 'k' by an angle 'theta'
    # v_rot = v * cos(theta) + (k x v) * sin(theta) + k * (k . v) * (1 - cos(theta))
    
    # Pre-calculate sine and cosine for the broadcasted rotation
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    # Cross product for the rotation
    cross_prod = np.cross(axis, v_ca_n)
    
    # Dot product for the rotation
    # We need to reshape for correct broadcasting: (L, 1) * (L, 3) -> (L, 3)
    dot_prod = np.sum(axis * v_ca_n, axis=-1, keepdims=True) * axis

    # Apply Rodrigues' formula
    v_ca_n_rotated = (v_ca_n * cos_angle) + \
                     (cross_prod * sin_angle) + \
                     (dot_prod * (1 - cos_angle))

    # The rotated vector now points from C-alpha to the C-beta position.
    # We add this vector to the C-alpha coordinates to get the final CB coordinates.
    CB_coords = CA_coords + v_ca_n_rotated
    
    return CB_coords


def find_pore_and_surf_residues(coords_CA, coords_N, coords_C):
    '''Assumes the coords pertain to Calpha coordinates of a beta-barrel protein whose pore-axis is aligned to the Z-axis. Now using idealized Cbeta positions
    and Calpha-Cbeta vector directions, find the pore-facing and surface-facing residue indices of the beta-barrel. Then take the intersection with intf_tmd and buried_tmd.'''
    
    # get projections of Calpha coordinates on the XY-plane
    xy_proj_CA = coords_CA[:, :2]
    # get the projections of the Cbeta coordinates on the XY-plane
    coords_CB = calculate_cbeta_coordinates(coords_N, coords_C, coords_CA)
    xy_proj_CB = coords_CB[:, :2]
    # Now get the dot product of the xy_proj_CA and (xy_proj_CB - xy_proj_CA) vectors
    dot_products = numpy.einsum('ij,ij->i', xy_proj_CA, (xy_proj_CB - xy_proj_CA))
    # pore-facing residues have negative dot products, surface-facing residues have positive dot products
    pore_facing = numpy.where(dot_products < 0)[0]
    surface_facing = numpy.where(dot_products > 0)[0]
    

    return pore_facing, surface_facing


def get_intf_buried_resi(coords, TMdepth, interf_frac, surface_facing):
    maxZ = numpy.max(coords, 0)[2]
    minZ = numpy.min(coords, 0)[2]

    span = maxZ - minZ

    if span > TMdepth:
        resi_loop = 1 + numpy.nonzero(numpy.where(coords[:,2]< minZ+TMdepth, coords[:,2], 0))[0]
        resi_tmd = numpy.array(resi_loop)
        maxZ = numpy.max(coords[resi_tmd-1,2])
        span = maxZ - minZ
    else:
        resi_tmd = numpy.array(list(range(1, len(coords)+1)))

    intf = (span*interf_frac)/2

    intf1_tmd = [x for x in numpy.where(coords[resi_tmd-1,2]<= minZ+2.0*intf, resi_tmd-1, 0) if x != 0.0]

    intf_tmd = intf1_tmd + [x for x in numpy.where(coords[resi_tmd-1,2]>= maxZ-intf, resi_tmd-1, 0) if x != 0.0]

    intf_coord = numpy.array([coords[i-1] for i in intf_tmd])
    maxZ = numpy.max(intf_coord, 0)[2]

    buriedtmd = [x for x in numpy.where(coords[resi_tmd-1,2]< maxZ, resi_tmd-1, 0) if x != 0.0 and x not in intf_tmd]

    labels = numpy.zeros(len(coords))
    # take the intersection of surface_facing with intf_tmd
    labels[numpy.intersect1d(surface_facing, intf_tmd)] = 1.0
    labels[numpy.intersect1d(surface_facing, buriedtmd)] = 2.0

    return list(labels)


tmdepth = float(sys.argv[1])
interf_frac = float(sys.argv[2])

for pdb_name in sys.argv[3:]:
    pdbPath = Path(pdb_name)
    pdbpath = pdbPath.resolve()
    pdbdict[f'{pdbpath}'] = []
    #--------------------------------------------------------------------------
    # compute principal axes
    #--------------------------------------------------------------------------
    # read pdb
    coord_ca, coord_n, coord_c, allcoord, pdbst = read_pdb_coord(pdb_name)

    # compute geometric center
    center = numpy.mean(allcoord, 0)

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

    moved_coord_ca = numpy.dot(coord_ca-center, rotmat.T)
    moved_coord_n = numpy.dot(coord_n-center, rotmat.T)
    moved_coord_c = numpy.dot(coord_c-center, rotmat.T)

    if moved_coord_ca[0][2] > 0.0:
        moved_coord_ca, moved_coord_c, moved_coord_n = invert_pdb(moved_coord_ca, moved_coord_c, moved_coord_n)
        
    _, surface_facing = find_pore_and_surf_residues(moved_coord_ca, moved_coord_n, moved_coord_c)

    intf_buried_dict[f'{pdbpath}'] = get_intf_buried_resi(moved_coord_ca, tmdepth, interf_frac, surface_facing)


with open(out_file_pdb, 'w') as fout:
    json.dump(pdbdict, fout)

with open(out_file_intf_buried, 'w') as fout:
    json.dump(intf_buried_dict, fout)
