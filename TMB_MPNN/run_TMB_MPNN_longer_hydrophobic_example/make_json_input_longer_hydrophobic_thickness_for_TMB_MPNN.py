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
    """
    Make rotation matrix that aligns vec to the z-axis.
    """
    target = numpy.array([0, 0, 1])

    cross_product = numpy.cross(vec, target)
    dot_product = numpy.dot(vec, target)

    # If vec is already aligned to z-axis
    if numpy.isclose(dot_product, 1.0):
        return numpy.identity(3)

    # If vec is opposite to z-axis
    if numpy.isclose(dot_product, -1.0):
        return numpy.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1],
        ])

    skew_matrix = numpy.array([
        [0, -cross_product[2], cross_product[1]],
        [cross_product[2], 0, -cross_product[0]],
        [-cross_product[1], cross_product[0], 0],
    ])

    rotation_matrix = (
        numpy.identity(3)
        + skew_matrix
        + numpy.matmul(skew_matrix, skew_matrix) / (1 + dot_product)
    )

    return rotation_matrix


def invert_pdb(coord1, coord2, coord3):
    """
    Flip barrel orientation.
    """
    rotmat = numpy.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, -1],
    ])

    coord1 = numpy.dot(coord1, rotmat.T)
    coord2 = numpy.dot(coord2, rotmat.T)
    coord3 = numpy.dot(coord3, rotmat.T)

    return coord1, coord2, coord3


def read_pdb_coord(pdb_name):
    """
    Read CA, N, C, and all-atom coordinates from a PDB.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pdb", pdb_name)

    ca_coords = []
    n_coords = []
    c_coords = []
    all_coords = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.has_id("CA"):
                    ca_coords.append(residue["CA"].coord)
                if residue.has_id("N"):
                    n_coords.append(residue["N"].coord)
                if residue.has_id("C"):
                    c_coords.append(residue["C"].coord)

                for atom in residue:
                    all_coords.append(atom.coord)

    return (
        numpy.array(ca_coords),
        numpy.array(n_coords),
        numpy.array(c_coords),
        numpy.array(all_coords),
        structure,
    )


def get_intf_buried_resi(coords, tmdepth, interf_frac):
    """
    Assign hydrophobic-thickness conditioning labels.

    Token mapping:
      1 = interface
      2 = buried

    No pore-facing/surface-facing filtering.
    No 0 token.
    """
    max_z = numpy.max(coords, axis=0)[2]
    min_z = numpy.min(coords, axis=0)[2]
    span = max_z - min_z

    # If structure is taller than tmdepth, use only min_z to min_z + tmdepth
    if span > tmdepth:
        resi_tmd_1based = 1 + numpy.nonzero(
            numpy.where(coords[:, 2] < min_z + tmdepth, coords[:, 2], 0)
        )[0]
        resi_tmd_1based = numpy.array(resi_tmd_1based)

        if len(resi_tmd_1based) == 0:
            # fallback: everything interface
            return [1.0] * len(coords)

        max_z = numpy.max(coords[resi_tmd_1based - 1, 2])
        span = max_z - min_z
    else:
        resi_tmd_1based = numpy.array(list(range(1, len(coords) + 1)))

    interface_width = (span * interf_frac) / 2.0

    # Bottom interface
    intf1_tmd_0based = [
        x for x in numpy.where(
            coords[resi_tmd_1based - 1, 2] <= min_z + 2.0 * interface_width,
            resi_tmd_1based - 1,
            -1,
        )
        if x != -1
    ]

    # Top interface
    intf2_tmd_0based = [
        x for x in numpy.where(
            coords[resi_tmd_1based - 1, 2] >= max_z - interface_width,
            resi_tmd_1based - 1,
            -1,
        )
        if x != -1
    ]

    intf_tmd_0based = sorted(set(intf1_tmd_0based + intf2_tmd_0based))

    if len(intf_tmd_0based) == 0:
        # fallback: everything interface
        return [1.0] * len(coords)

    intf_coord = numpy.array([coords[i] for i in intf_tmd_0based])
    max_z_intf = numpy.max(intf_coord, axis=0)[2]

    buriedtmd_0based = [
        x for x in numpy.where(
            coords[resi_tmd_1based - 1, 2] < max_z_intf,
            resi_tmd_1based - 1,
            -1,
        )
        if x != -1 and x not in intf_tmd_0based
    ]

    # Hydrophobic thickness version:
    # default everything to interface = 1
    # then set buried residues to 2
    labels = numpy.ones(len(coords), dtype=float)
    labels[buriedtmd_0based] = 2.0

    return list(labels)


if len(sys.argv) < 4:
    print(
        "Usage: python make_json_input_longer_hydrophobic_thickness_for_TMB_MPNN.py "
        "<tmdepth> <interf_frac> <pdb1> [pdb2 ...]"
    )
    sys.exit(1)


tmdepth = float(sys.argv[1])
interf_frac = float(sys.argv[2])

for pdb_name in sys.argv[3:]:
    pdb_path = Path(pdb_name).resolve()

    pdbdict[f"{pdb_path}"] = []

    coord_ca, coord_n, coord_c, allcoord, pdb_structure = read_pdb_coord(pdb_name)

    if len(coord_ca) == 0:
        print(f"Warning: no CA atoms found in {pdb_name}. Skipping.")
        continue

    center = numpy.mean(allcoord, axis=0)

    # Use modified CA coordinates for principal-axis calculation
    coord = mod_coords(coord_ca - center)

    inertia = numpy.dot(coord.transpose(), coord)
    e_values, e_vectors = numpy.linalg.eig(inertia)

    order = numpy.argsort(e_values)
    eval3, eval2, eval1 = e_values[order]
    axis3, axis2, axis1 = e_vectors[:, order].transpose()

    rotmat = make_rotmat(axis3 / numpy.linalg.norm(axis3))

    moved_coord_ca = numpy.dot(coord_ca - center, rotmat.T)
    moved_coord_n = numpy.dot(coord_n - center, rotmat.T)
    moved_coord_c = numpy.dot(coord_c - center, rotmat.T)

    if moved_coord_ca[0][2] > 0.0:
        moved_coord_ca, moved_coord_c, moved_coord_n = invert_pdb(
            moved_coord_ca,
            moved_coord_c,
            moved_coord_n,
        )

    intf_buried_dict[f"{pdb_path}"] = get_intf_buried_resi(
        moved_coord_ca,
        tmdepth,
        interf_frac,
    )


with open(out_file_pdb, "w") as fout:
    json.dump(pdbdict, fout)

with open(out_file_intf_buried, "w") as fout:
    json.dump(intf_buried_dict, fout)

print(f"Wrote {out_file_pdb}")
print(f"Wrote {out_file_intf_buried}")