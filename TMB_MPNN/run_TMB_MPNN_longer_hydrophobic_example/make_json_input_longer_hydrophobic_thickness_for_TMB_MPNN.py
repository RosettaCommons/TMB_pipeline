#!/~ENV_DIR/modulus_latest.sif
import sys
import json
from pathlib import Path
import numpy
import numpy as np
from Bio.PDB import PDBParser

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
        y = 1 / x**3
    return y


def mod_coords(coords):
    factor = []
    norm = numpy.linalg.norm(coords, axis=1)
    for r in norm:
        factor.append(scale_func(r))

    new_coords = (coords.T * numpy.array(factor)).T
    return new_coords


def make_rotmat(vec):
    # Calculate the cross product and dot product of the input and target vectors
    cross_product = numpy.cross(vec, numpy.array([0, 0, 1]))
    dot_product = numpy.dot(vec, numpy.array([0, 0, 1]))

    # Handle the rare case where vec is exactly opposite to z-axis
    if numpy.isclose(dot_product, -1.0):
        return numpy.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

    # Calculate the skew-symmetric matrix
    skew_matrix = numpy.array([
        [0, -cross_product[2], cross_product[1]],
        [cross_product[2], 0, -cross_product[0]],
        [-cross_product[1], cross_product[0], 0]
    ])

    # Calculate the rotation matrix
    rotation_matrix = (
        numpy.identity(3)
        + skew_matrix
        + numpy.matmul(skew_matrix, skew_matrix) / (1 + dot_product)
    )

    return rotation_matrix


def invert_pdb(coord1, coord2, coord3):
    rotmat = numpy.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    coord1 = numpy.dot(coord1, rotmat.T)
    coord2 = numpy.dot(coord2, rotmat.T)
    coord3 = numpy.dot(coord3, rotmat.T)
    return coord1, coord2, coord3


def read_pdb_coord(pdb_name):
    """
    Reads atomic coordinates of C-alpha, N, and C atoms from a .pdb file.

    Parameters
    ----------
    pdb_name : str
        Name of pdb file.

    Returns
    -------
    tuple
        CA, N, C, all-atom coordinates, and structure object
    """
    p = PDBParser(QUIET=True)
    st = p.get_structure("pdb", pdb_name)

    coordinates = []
    N_coords = []
    C_coords = []
    ca = []

    for model in st:
        for chain in model:
            for residue in chain:
                if residue.has_id("CA"):
                    ca.append(residue["CA"].coord)
                if residue.has_id("N"):
                    N_coords.append(residue["N"].coord)
                if residue.has_id("C"):
                    C_coords.append(residue["C"].coord)
                for atom in residue:
                    coordinates.append(atom.coord)

    return (
        numpy.array(ca),
        numpy.array(N_coords),
        numpy.array(C_coords),
        numpy.array(coordinates),
        st,
    )


def get_intf_buried_resi(coords, TMdepth, interf_frac):
    """
    Label residues by membrane position only.

    Token mapping:
      0 = outside labeled TMD region
      1 = interface
      2 = buried

    Notes
    -----
    This follows the original script's z-axis logic, but removes all
    pore-facing vs surface-facing filtering.
    """
    maxZ = numpy.max(coords, axis=0)[2]
    minZ = numpy.min(coords, axis=0)[2]

    span = maxZ - minZ

    # Original behavior: if structure is taller than TMdepth,
    # keep only residues within minZ to minZ + TMdepth
    if span > TMdepth:
        resi_tmd_1based = 1 + numpy.nonzero(
            numpy.where(coords[:, 2] < minZ + TMdepth, coords[:, 2], 0)
        )[0]
        resi_tmd_1based = numpy.array(resi_tmd_1based)

        if len(resi_tmd_1based) == 0:
            return [0.0] * len(coords)

        maxZ = numpy.max(coords[resi_tmd_1based - 1, 2])
        span = maxZ - minZ
    else:
        resi_tmd_1based = numpy.array(list(range(1, len(coords) + 1)))

    intf = (span * interf_frac) / 2.0

    # Keep original interface logic
    intf1_tmd_0based = [
        x for x in numpy.where(
            coords[resi_tmd_1based - 1, 2] <= minZ + 2.0 * intf,
            resi_tmd_1based - 1,
            -1
        )[0]
        if x != -1
    ]

    intf2_tmd_0based = [
        x for x in numpy.where(
            coords[resi_tmd_1based - 1, 2] >= maxZ - intf,
            resi_tmd_1based - 1,
            -1
        )[0]
        if x != -1
    ]

    intf_tmd_0based = sorted(set(intf1_tmd_0based + intf2_tmd_0based))

    if len(intf_tmd_0based) == 0:
        return [0.0] * len(coords)

    intf_coord = numpy.array([coords[i] for i in intf_tmd_0based])
    maxZ_intf = numpy.max(intf_coord, axis=0)[2]

    buriedtmd_0based = [
        x for x in numpy.where(
            coords[resi_tmd_1based - 1, 2] < maxZ_intf,
            resi_tmd_1based - 1,
            -1
        )[0]
        if x != -1 and x not in intf_tmd_0based
    ]

    labels = numpy.zeros(len(coords), dtype=float)
    labels[intf_tmd_0based] = 1.0
    labels[buriedtmd_0based] = 2.0

    return list(labels)


tmdepth = float(sys.argv[1])
interf_frac = float(sys.argv[2])

for pdb_name in sys.argv[3:]:
    pdbPath = Path(pdb_name)
    pdbpath = pdbPath.resolve()
    pdbdict[f"{pdbpath}"] = []

    # read pdb
    coord_ca, coord_n, coord_c, allcoord, pdbst = read_pdb_coord(pdb_name)

    # compute geometric center
    center = numpy.mean(allcoord, axis=0)

    # center with geometric center
    coord = mod_coords(coord_ca - center)

    # compute principal axis matrix
    inertia = numpy.dot(coord.transpose(), coord)
    e_values, e_vectors = numpy.linalg.eig(inertia)

    # order eigen values and eigen vectors
    order = numpy.argsort(e_values)
    eval3, eval2, eval1 = e_values[order]
    axis3, axis2, axis1 = e_vectors[:, order].transpose()

    rotmat = make_rotmat(axis3 / numpy.linalg.norm(axis3))

    moved_coord_ca = numpy.dot(coord_ca - center, rotmat.T)
    moved_coord_n = numpy.dot(coord_n - center, rotmat.T)
    moved_coord_c = numpy.dot(coord_c - center, rotmat.T)

    if moved_coord_ca[0][2] > 0.0:
        moved_coord_ca, moved_coord_c, moved_coord_n = invert_pdb(
            moved_coord_ca, moved_coord_c, moved_coord_n
        )

    intf_buried_dict[f"{pdbpath}"] = get_intf_buried_resi(
        moved_coord_ca, tmdepth, interf_frac
    )

with open(out_file_pdb, "w") as fout:
    json.dump(pdbdict, fout)

with open(out_file_intf_buried, "w") as fout:
    json.dump(intf_buried_dict, fout)
