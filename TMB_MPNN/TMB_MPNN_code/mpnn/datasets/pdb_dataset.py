import itertools
import os

import numpy as np
import torch

from .cifutils import CIFParser
from .residue_mapping import restype_1to3


class PDBDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cif_path,
        parser_exclude_res_list=[],
        atom_list_to_save=["N", "CA", "C", "O"],
        parse_only_DNA_RNA=0,
        protein_backbone_occ_cutoff=0.8,
        protein_side_chain_occ_cutoff=0.5,
        excluded_elements=[1],
        ligand_occ_cutoff=0.01,
        num_nn=25,
        parse_only_proteins=0,
    ):

        self.num_nn = num_nn
        self.ligand_occ_cutoff = ligand_occ_cutoff
        self.excluded_elements = excluded_elements
        self.protein_backbone_occ_cutoff = protein_backbone_occ_cutoff
        self.protein_side_chain_occ_cutoff = protein_side_chain_occ_cutoff
        self.parse_only_DNA_RNA = parse_only_DNA_RNA
        self.parse_only_proteins = parse_only_proteins
        if self.parse_only_DNA_RNA:
            self.exclude_v_list = ["nonpoly"]
        else:
            self.exclude_v_list = []
        self.atom_list_to_save = atom_list_to_save
        self.parser = CIFParser(skip_res=parser_exclude_res_list)
        self.cif_path = cif_path

        self.restype_3to1 = {v: k for k, v in restype_1to3.items()}

        self.alphabet = list("ACDEFGHIKLMNPQRSTVWYX")
        self.alphabet_dict = dict(zip(self.alphabet, range(len(self.alphabet))))

        self.bb_idx_list = []
        self.backbone_list = ["N", "CA", "C", "O"]
        for i, atom in enumerate(self.atom_list_to_save):
            if atom in self.backbone_list:
                self.bb_idx_list.append(i)
        
        self.sample_num = 0  # will update by hook

    def __getitem__(self, index):
        """
        index = [[("pdb_id", "biounit_id"), ("pdb_id", "biounit_id")]]
        """
        x = [self.loader(pdb_tuple[0], pdb_tuple[1]) for pdb_tuple in index[0]]
        return x

    def __len__(self):
        return self.sample_num

    def loader(self, pdb_id, assembly_id):
        assert len(pdb_id) == 4
        subfolder = pdb_id[1:3]
        cache_folder = f'/net/scratch/faxue/mpnn_preprocessed_TrRosetta/{subfolder}'
        os.makedirs(cache_folder, exist_ok=True)
        cache_path = f'{cache_folder}/{pdb_id}_{assembly_id}.pth'
        if os.path.exists(cache_path):
            out_dict = torch.load(cache_path)
            return (out_dict, out_dict["protein_L"])

        # processing
        try:
            chains, asmb, covalei, meta = self.parser.parse(self.cif_path + f"/{pdb_id[1:3]}/{pdb_id}.cif.gz")
        except:
            print("bad_id: ", pdb_id)
            return ("pass", "pass")
        # ------------------proteins vs not--------------------
        polypeptide_letter_list = []
        ligand_letter_list = []
        for chain_letter, v in chains.items():
            if v.type == "polypeptide(L)":
                polypeptide_letter_list.append(chain_letter)
            else:
                if v.type not in self.exclude_v_list:
                    ligand_letter_list.append(chain_letter)

        if self.parse_only_proteins:
            ligand_letter_list = []

        atom_list_to_save_dict = {atom: idx for idx, atom in enumerate(self.atom_list_to_save)}

        poly_chain_dict = {}
        extra_ligand_chain_dict = {}

        for poly_chain in polypeptide_letter_list:
            extra_ligand_xyz_list = []
            extra_ligand_element_list = []

            atoms = list(chains[poly_chain].atoms.keys())
            L = len(set([a[1] for a in atoms]))
            # L = len(list(set([a[1] for a in list(chains[poly_chain].atoms.keys())])))
            xyz = np.zeros([L, len(self.atom_list_to_save), 3], dtype=np.float32)
            occ = np.zeros([L, len(self.atom_list_to_save)], dtype=np.float32)
            residue_idx = -100 * np.ones([L], dtype=np.int32)
            raw_sequence = L * ["UNK"]

            for c, (res_id, atom_tuple) in enumerate(   # c is the index of the residue
                itertools.groupby(atoms, lambda x: x[1])
            ):
                for atom_key in atom_tuple:     # ('A', '4', 'MSE', 'N')
                    atom = chains[poly_chain].atoms[atom_key]
                    if atom_key[2] not in list(self.restype_3to1):
                        if atom.element not in self.excluded_elements and atom.occ > self.ligand_occ_cutoff:
                            extra_ligand_xyz_list.append(np.array(atom.xyz))
                            extra_ligand_element_list.append(np.array(atom.element))

                    else:
                        if atom_key[3] in self.atom_list_to_save:
                            atom_idx = atom_list_to_save_dict[atom_key[3]]
                            xyz[c, atom_idx, :] = np.array(atom.xyz)
                            occ[c, atom_idx] = np.array(atom.occ)
                            if atom_key[3] == "CA":
                                raw_sequence[c] = atom_key[2]
                                residue_idx[c] = int(atom_key[1])

            if len(extra_ligand_xyz_list) > 0:
                extra_ligand_chain_dict[poly_chain] = {
                    "xyz": np.stack(extra_ligand_xyz_list),
                    "element": np.stack(extra_ligand_element_list)
                }

            mask_occ = occ > self.protein_backbone_occ_cutoff
            mask_occ = (
                mask_occ[:, self.bb_idx_list[0]]
                * mask_occ[:, self.bb_idx_list[1]]
                * mask_occ[:, self.bb_idx_list[2]]
                * mask_occ[:, self.bb_idx_list[3]]
            ).astype(bool)

            poly_chain_dict[poly_chain] = {
                "xyz": xyz[mask_occ],
                "occ": occ[mask_occ],
                "seq": [AA for i, AA in enumerate(raw_sequence) if mask_occ[i]],
                "residue_idx": residue_idx[mask_occ]
            }

        ligand_chain_dict = {}
        for letter in ligand_letter_list:
            xyz_list = []
            element_list = []
            for k, v in chains[letter].atoms.items():
                if v.element not in self.excluded_elements:
                    if v.occ > self.ligand_occ_cutoff:
                        xyz_list.append(np.array(v.xyz))
                        element_list.append(np.array(v.element))

            if len(xyz_list) > 0:
                ligand_chain_dict[letter] = {
                    "xyz": np.stack(xyz_list),
                    "element": np.stack(element_list)
                }

        protein_L = 0
        Cb_list = []
        Y_list = []
        Y_t_list = []
        X_list = []
        X_m_list = []
        S_list = []
        R_idx_list = []
        chain_labels_list = []
        chain_multi = 0

        poly_chain_list = list(poly_chain_dict.keys())

        if assembly_id not in asmb:
            return ("pass", "pass")

        for item in asmb[assembly_id]:
            chain_id = item[0]
            rotation_matrix = item[1][:3, :3]
            translation = item[1][:3, 3]

            if chain_id in poly_chain_list:
                chain_data = poly_chain_dict[chain_id]
                tmp_residue_idx = chain_data["residue_idx"]
                xyz = chain_data["xyz"]
                xyz = np.einsum("ij,raj->rai", rotation_matrix, xyz) + translation
                protein_L += xyz.shape[0]
                N = xyz[:, self.bb_idx_list[0], :]
                Ca = xyz[:, self.bb_idx_list[1], :]
                C = xyz[:, self.bb_idx_list[2], :]

                b = Ca - N
                c = C - Ca
                a = np.cross(b, c, axis=-1)
                Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca
                
                Cb_list.append(Cb)
                X_list.append(xyz)
                X_m = (chain_data["occ"] > self.protein_side_chain_occ_cutoff).astype(bool)
                X_m_list.append(X_m)

                seq_one_letter = [
                    self.restype_3to1[AA] if AA in list(self.restype_3to1) else "X"
                    for AA in chain_data["seq"]
                ]
                seq_int = [
                    self.alphabet_dict[AA] if AA in self.alphabet else self.alphabet_dict["X"] for AA in seq_one_letter
                ]

                S_list.append(np.array(seq_int, dtype=np.int32))
                R_idx_list.append(tmp_residue_idx)
                chain_labels_list.append(chain_multi * np.ones_like(tmp_residue_idx, dtype=np.int32))
                chain_multi += 1

                if chain_id in extra_ligand_chain_dict:
                    xyz = extra_ligand_chain_dict[chain_id]["xyz"]
                    xyz = np.einsum("ij,rj->ri", rotation_matrix, xyz) + translation
                    Y_list.append(xyz)
                    Y_t_list.append(extra_ligand_chain_dict[chain_id]["element"])

            elif chain_id in ligand_chain_dict:
                xyz = ligand_chain_dict[chain_id]["xyz"]
                xyz = np.einsum("ij,rj->ri", rotation_matrix, xyz) + translation
                Y_list.append(xyz)
                Y_t_list.append(ligand_chain_dict[chain_id]["element"])

        # Pick closest neighbors per residue using virtual CB distance to context atoms
        Cb_stack = np.concatenate(Cb_list, axis=0)  # [L, 3]
        if len(Y_list) > 0:
            Y_stack = np.concatenate(Y_list, axis=0)  # [M, 3]
            Y_t_stack = np.concatenate(Y_t_list, axis=0)  # [M]

            D = np.sum((Cb_stack[:, None, :] - Y_stack[None, :, :]) ** 2, axis=-1)  # [L, M]
            closest_idx = np.argsort(D, axis=-1)[:, : self.num_nn]
            ligand_xyz_closest = np.take_along_axis(
                np.repeat(Y_stack[None, :, :], Cb_stack.shape[0], 0), np.repeat(closest_idx[:, :, None], 3, -1), axis=1
            )
            ligand_element_closest = np.take_along_axis(
                np.repeat(Y_t_stack[None, :], Cb_stack.shape[0], 0), closest_idx, axis=1
            )
        else:
            ligand_xyz_closest = np.zeros([Cb_stack.shape[0], 1, 3], dtype=np.float32)
            ligand_element_closest = np.zeros([Cb_stack.shape[0], 1], dtype=np.float32)

        X_stack = np.concatenate(X_list, axis=0)  # [L, num_atoms, 3]
        X_m_stack = np.concatenate(X_m_list, axis=0)  # [L, num_atoms]
        S_stack = np.concatenate(S_list, axis=0)  # [L]
        R_idx_stack = np.concatenate(R_idx_list, axis=0)  # [L]
        chain_labels_stack = np.concatenate(chain_labels_list, axis=0)  # [L]

        out_dict = {
            "Y": ligand_xyz_closest,
            "Y_t": ligand_element_closest,
            "X": X_stack,
            "X_m": X_m_stack,
            "S": S_stack,
            "R_idx": R_idx_stack,
            "chain_labels": chain_labels_stack,
            "protein_L": protein_L
        }

        # save cache
        torch.save(out_dict, cache_path)

        return (out_dict, protein_L)
