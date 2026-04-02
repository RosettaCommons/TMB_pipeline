import copy
import itertools
import os
import random
from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import biotite.structure as struc
import numpy as np
import periodictable
import torch
from cifutils.constants import CRYSTALLIZATION_AIDS
from cifutils.transforms.categories import get_chain_info_from_category
from cifutils.utils.io_utils import read_any
from cifutils.utils.non_rcsb_utils import infer_chain_info_from_atom_array

from .cifutils import CIFParser
from .fused_data_utils import get_nearest_neighbours, parse_PDB
from .residue_mapping import (
    BACKBONE_ATOMS,
    PROTEIN_ALL_ATOMS,
    restype_3to1,
    restype_INTtoSTR,
    restype_STRtoINT,
)


class BaseTransform(metaclass=ABCMeta):
    """Base class for all transformations."""

    def __call__(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:

        return self.transform(results)

    @abstractmethod
    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        """The transform function. All subclass of BaseTransform should
        override this method.

        This function takes the result dict as the input, and can add new
        items to the dict or modify existing items in the dict. And the result
        dict will be returned in the end, which allows to concate multiple
        transforms into a pipeline.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """


class Compose:
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict, callable], optional): Sequence of transform
            object to be composed.
    """

    def __init__(self, transforms: Optional[Sequence[Union[dict, Callable]]]):
        self.transforms: List[Callable] = []

        if transforms is None:
            transforms = []

        for transform in transforms:
            if callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(
                    f"transform must be a callable object or dict, " f"but got {type(transform)}"
                )

    def __call__(self, data: dict) -> Optional[dict]:
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """
        for t in self.transforms:
            data = t(data)
            # The transform will return None when it failed to load images or
            # cannot find suitable augmentation parameters to augment the data.
            # Here we simply return None if the transform returns None and the
            # dataset will handle it by randomly selecting another data sample.
            if data is None:
                return None
        return data

    def __repr__(self):
        """Print ``self.transforms`` in sequence.

        Returns:
            str: Formatted string.
        """
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class LoadFile(BaseTransform):

    def transform(self, results):
        cif_path = results["cif_path"]
        file_obj = read_any(cif_path)
        results["file_obj"] = file_obj
        return results


class GetStructure(BaseTransform):
    """Directly load the structure, suitable for PDB file"""

    def __init__(self, random_select_model=False):
        """
        Args:
            random_select_model: whether to randomly select a model, will return the first model if False
        """
        self.random_select_model = random_select_model

    def _choose_model(self, atom_array):
        if len(atom_array.shape) > 1:
            if self.random_select_model:
                i = random.randint(0, atom_array.shape[0] - 1)
                atom_array = atom_array[i]
            else:
                atom_array = atom_array[0]
        return atom_array

    def transform(self, results):
        cif_path = results["cif_path"]
        file_obj = results["file_obj"]

        filename = cif_path.split("/")[-1]
        if ".cif" in filename:
            atom_array = struc.io.pdbx.get_structure(
                file_obj,
                extra_fields=["occupancy", "b_factor"],
            )
        elif ".pdb" in filename:
            atom_array = struc.io.pdb.get_structure(
                file_obj, extra_fields=["occupancy", "b_factor"]
            )
        else:
            raise ValueError(f"Unknown file type: {filename}")

        atom_array = self._choose_model(atom_array)

        results["atom_array"] = atom_array
        return results


class GetAssembly(BaseTransform):
    """Get the specific assembly from the cif file"""

    def transform(self, results):
        cif_path = results["cif_path"]
        assembly_id = results["assembly_id"]
        file_obj = results["file_obj"]

        filename = cif_path.split("/")[-1]
        if ".cif" in filename:
            func = partial(struc.io.pdbx.get_assembly, use_author_fields=False)
        elif ".pdb" in filename:
            func = struc.io.pdb.get_assembly
        else:
            raise ValueError(f"Unknown file type: {filename}")
        atom_array = func(
            file_obj,
            assembly_id,
            extra_fields=["occupancy"],
        )

        if len(atom_array.shape) > 1:
            assert (
                atom_array.shape[0] == 1
            ), f"Multiple models ({atom_array.shape}) found in {cif_path}"
            atom_array = atom_array[0]

        results["atom_array"] = atom_array
        return results


class ExcludeByResName(BaseTransform):
    """Exclude residues by res_name"""

    def __init__(self, exclude_res_list: List[str]):
        exclude_by_default = CRYSTALLIZATION_AIDS + ["HOH", "NA", "CL", "K", "BR", "UNX"]
        self.exclude_res_list = list(set(exclude_res_list + exclude_by_default))

    def transform(self, results):
        atom_array = results["atom_array"]
        mask = np.isin(atom_array.res_name, self.exclude_res_list, invert=True)
        results["atom_array"] = atom_array[mask]
        return results


class GetChainInfo(BaseTransform):
    """Get chain info from the atom_array"""

    def transform(self, results):

        atom_array = results["atom_array"]
        try:
            chain_info = get_chain_info_from_category(
                results["file_obj"].block, atom_array
            )  # TODO: this will never succeed: Annotation category 'label_entity_id' is not existing
        except Exception as e:
            print(f"Error in getting chain info: {e}")
            chain_info = infer_chain_info_from_atom_array(atom_array)
        results["chain_info"] = chain_info
        return results


class SeperateProteinLigand(BaseTransform):
    """Extract ligand from the atom_array"""

    def __init__(
        self,
        atoms_to_load="all",
        excluded_elements=[1],
        protein_backbone_occ_cutoff=0.8,
        protein_side_chain_occ_cutoff=0.5,
        ligand_occ_cutoff=0.01,
    ):
        assert atoms_to_load in ["all", "backbone"]
        self.atom_list_to_load = atoms_to_load
        self.excluded_elements = excluded_elements
        self.protein_backbone_occ_cutoff = protein_backbone_occ_cutoff
        self.protein_side_chain_occ_cutoff = protein_side_chain_occ_cutoff
        self.ligand_occ_cutoff = ligand_occ_cutoff

    def transform(self, results):
        atom_array = results["atom_array"]

        chain_info = results["chain_info"]

        protein_chain_list = []
        ligand_chain_list = []

        for chain_letter, v in chain_info.items():
            curr_chain_array = atom_array[atom_array.chain_id == chain_letter]
            chain_backbone_mask = struc.filter_canonical_amino_acids(curr_chain_array)
            # TODO: how to determine if it is a ligand
            if v["type"].lower() == "polypeptide(l)" and chain_backbone_mask.sum() / 4 > 10:
                protein_chain_list.append(chain_letter)
            else:
                ligand_chain_list.append(chain_letter)

        poly_chain_dict = {}
        # ligand atoms that are in the protein chain (e.g. unstandard amino acids)
        extra_ligand_chain_dict = {}

        if self.atom_list_to_load == "backbone":
            atom_list_to_save = BACKBONE_ATOMS
        elif self.atom_list_to_load == "all":
            atom_list_to_save = PROTEIN_ALL_ATOMS
        else:
            raise ValueError(f"Unknown atoms_to_load: {self.atom_list_to_load}")

        for chain_id in protein_chain_list:
            extra_ligand_xyz_list = []
            extra_ligand_element_list = []

            atoms = atom_array[atom_array.chain_id == chain_id]
            L = struc.get_residue_count(atoms)
            xyz = np.zeros([L, len(atom_list_to_save), 3], dtype=np.float32)
            occ = np.zeros([L, len(atom_list_to_save)], dtype=np.float32)
            residue_idx = -100 * np.ones([L], dtype=np.int32)
            raw_sequence = L * ["UNK"]

            for c, (res_id, atom_tuple) in enumerate(itertools.groupby(atoms, lambda x: x.res_id)):
                for atom in atom_tuple:
                    if atom.res_name not in restype_3to1:
                        if (
                            atom.element not in self.excluded_elements
                            and atom.occupancy > self.ligand_occ_cutoff
                        ):
                            extra_ligand_xyz_list.append(atom.coord)
                            extra_ligand_element_list.append(np.array(atom.element))
                    else:  # standard amino acid
                        if atom.atom_name in atom_list_to_save:
                            atom_idx = atom_list_to_save.index(atom.atom_name)
                            xyz[c, atom_idx, :] = atom.coord
                            occ[c, atom_idx] = atom.occupancy
                            if atom.atom_name == "CA":
                                raw_sequence[c] = atom.res_name
                                residue_idx[c] = int(atom.res_id)

            if len(extra_ligand_xyz_list) > 0:
                extra_ligand_chain_dict[chain_id] = {
                    "xyz": np.stack(extra_ligand_xyz_list),
                    "element": np.stack(extra_ligand_element_list),
                }

            mask_occ = occ > self.protein_backbone_occ_cutoff
            mask_occ = mask_occ[:, :4].sum(axis=1) == 4  # all backbone atoms

            poly_chain_dict[chain_id] = {
                "xyz": xyz[mask_occ],
                "occ": occ[mask_occ],
                "seq": [res_name for i, res_name in enumerate(raw_sequence) if mask_occ[i]],
                "residue_idx": residue_idx[mask_occ],
            }
            poly_chain_dict[chain_id]["side_chain_mask"] = (
                poly_chain_dict[chain_id]["occ"] > self.protein_side_chain_occ_cutoff
            )

        ligand_chain_dict = {}
        for chain_id in ligand_chain_list:
            xyz_list = []
            element_list = []
            type_list = []
            for atom in atom_array[atom_array.chain_id == chain_id]:
                if atom.element not in self.excluded_elements:
                    if atom.occupancy > self.ligand_occ_cutoff:
                        xyz_list.append(atom.coord)
                        element_list.append(np.array(atom.element))

            if len(xyz_list) > 0:
                ligand_chain_dict[chain_id] = {
                    "xyz": np.stack(xyz_list),
                    "element": np.stack(element_list),
                    "res_type": type_list,
                }

        results["poly_chain_dict"] = poly_chain_dict
        results["ligand_chain_dict"] = ligand_chain_dict
        results["extra_ligand_chain_dict"] = extra_ligand_chain_dict

        return results


class BuildMPNNInput(BaseTransform):
    def __init__(self, lig_nn_num=25, lig_cutoff_for_score=8.0):
        """
        Args:
            lig_nn_num: number of nearest ligand atoms (to Cb) to consider
        """
        self.lig_nn_num = lig_nn_num
        self.lig_cutoff_for_score = lig_cutoff_for_score

    def transform(self, results):
        poly_chain_dict = results["poly_chain_dict"]
        ligand_chain_dict = results["ligand_chain_dict"]
        extra_ligand_chain_dict = results["extra_ligand_chain_dict"]

        Cb_list = []
        X_list = []
        side_chain_mask_list = []
        seq_int_list = []
        res_idx_list = []
        chain_labels_list = []
        chain_letters_list = []

        lig_coords = []
        lig_elements = []

        chain_list = np.unique(results["atom_array"].chain_id)
        for chain_idx, chain_id in enumerate(chain_list):
            if chain_id in poly_chain_dict:
                chain_data = poly_chain_dict[chain_id]
                tmp_residue_idx = chain_data["residue_idx"]
                xyz = chain_data["xyz"]  # TODO: do we need to transform the coordinates?
                N = xyz[:, 0, :]
                CA = xyz[:, 1, :]
                C = xyz[:, 2, :]

                b = CA - N
                c = C - CA
                a = np.cross(b, c, axis=-1)
                CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA

                Cb_list.append(CB)
                X_list.append(xyz)
                side_chain_mask_list.append(chain_data["side_chain_mask"])

                seq_one_letter = [restype_3to1[AA] for AA in chain_data["seq"]]
                seq_int = [restype_STRtoINT[AA] for AA in seq_one_letter]
                seq_int_list.append(seq_int)
                res_idx_list.append(tmp_residue_idx)
                chain_labels_list.append(chain_idx * np.ones_like(tmp_residue_idx, dtype=np.int32))
                chain_letters_list.extend([chain_id] * len(tmp_residue_idx))

                if chain_id in extra_ligand_chain_dict:
                    xyz = extra_ligand_chain_dict[chain_id]["xyz"]
                    lig_coords.append(xyz)
                    lig_elements.append(extra_ligand_chain_dict[chain_id]["element"])

            elif chain_id in ligand_chain_dict:
                xyz = ligand_chain_dict[chain_id]["xyz"]
                lig_coords.append(xyz)
                lig_elements.append(ligand_chain_dict[chain_id]["element"])

        if len(Cb_list) > 0:
            Cb_stack = np.concatenate(Cb_list, axis=0)
            X_stack = np.concatenate(X_list, axis=0)
            side_chain_mask_stack = np.concatenate(side_chain_mask_list, axis=0)
            seq_int_stack = np.concatenate(seq_int_list, axis=0)
            res_idx_stack = np.concatenate(res_idx_list, axis=0)
            chain_labels_stack = np.concatenate(chain_labels_list, axis=0)
        else:
            Cb_stack = np.zeros([0, 3], dtype=np.float32)
            X_stack = np.zeros([0, 37, 3], dtype=np.float32)
            side_chain_mask_stack = np.zeros([0, 37], dtype=bool)
            seq_int_stack = np.zeros([0], dtype=np.int32)
            res_idx_stack = np.zeros([0], dtype=np.int32)
            chain_labels_stack = np.zeros([0], dtype=np.int32)

        if len(lig_coords) > 0:
            lig_coords = np.concatenate(lig_coords, axis=0)
            lig_elements = np.concatenate(lig_elements, axis=0)
            try:
                lig_elements = np.array(
                    [periodictable.__dict__[i.capitalize()].number for i in lig_elements],
                    dtype=np.int64,
                )
            except KeyError:
                Warning(f"Unknown element in ligand: {results['cif_path']}")
                return None

            D = np.sum((Cb_stack[:, None, :] - lig_coords[None, :, :]) ** 2, axis=-1)
            closest_idx = np.argsort(D, axis=-1)[:, : self.lig_nn_num]
            ligand_xyz_closest = np.take_along_axis(
                np.repeat(lig_coords[None, :, :], Cb_stack.shape[0], 0),
                np.repeat(closest_idx[:, :, None], 3, -1),
                axis=1,
            )
            ligand_element_closest = np.take_along_axis(
                np.repeat(lig_elements[None, :], Cb_stack.shape[0], 0), closest_idx, axis=1
            )
            D_XY = np.sqrt(np.sum((Cb_stack - ligand_xyz_closest[:, 0, :]) ** 2, axis=-1))
            Y_m = (ligand_element_closest != 1) & (ligand_element_closest != 0)
            mask_XY = (D_XY < self.lig_cutoff_for_score) * Y_m[:, 0]  # mask should be all True
        else:
            lig_coords = np.zeros([0, 3], dtype=np.float32)
            lig_elements = np.zeros([0], dtype=np.float32)
            ligand_xyz_closest = np.zeros(
                [Cb_stack.shape[0], self.lig_nn_num, 3], dtype=np.float32
            )
            ligand_element_closest = np.zeros([Cb_stack.shape[0], self.lig_nn_num], dtype=np.int64)
            # D_XY = np.full([Cb_stack.shape[0], 0], np.inf, dtype=np.float32)
            # Y_m = (ligand_element_closest != 1) & (ligand_element_closest != 0)
            Y_m = np.full((Cb_stack.shape[0], self.lig_nn_num), False, dtype=bool)
            mask_XY = np.full(Cb_stack.shape[0], False, dtype=bool)

        out_dict = dict(
            X=X_stack,
            X_m=side_chain_mask_stack,
            Y=ligand_xyz_closest,
            Y_t=ligand_element_closest,
            Y_coords=lig_coords,
            S=seq_int_stack,
            seq=[restype_INTtoSTR[i] for i in seq_int_stack],
            R_idx=res_idx_stack,
            chain_labels=chain_labels_stack,
            # for inference
            chain_letters=chain_letters_list,
            ins_codes=results["atom_array"][
                results["atom_array"].atom_name == "CA"
            ].ins_code.tolist(),
            mask=np.ones([X_stack.shape[0]], dtype=bool),
            Y_m=Y_m,
            mask_XY=mask_XY,
        )
        for key in out_dict:
            assert key not in results, f"Key {key} already exists in results"
        results.update(out_dict)
        return results


class BuildInferenceInput(BaseTransform):
    def transform(self, results: Dict) -> Dict | Tuple[List] | None:

        # renumber the residue index
        R_idx_original = copy.deepcopy(results["R_idx"])
        R_idx_list = []
        count = 0
        R_idx_prev = -100000
        for R_idx in list(results["R_idx"]):
            if R_idx_prev == R_idx:
                count += 1
            R_idx_list.append(R_idx + count)
            R_idx_prev = R_idx
        R_idx_renumbered = np.array(R_idx_list, dtype=np.int64)
        results["R_idx"] = R_idx_renumbered
        results["R_idx_original"] = np.array(R_idx_original, dtype=np.int64)
        return results


class RandomSelectAssembly(BaseTransform):

    def transform(self, results: Dict) -> Dict | Tuple[List] | None:
        """
        Required keys: "assembly_len_dict"
        Added keys: "assembly_id"
        """
        assembly_ids = list(results["assembly_len_dict"].keys())
        assembly_id = np.random.choice(assembly_ids)
        results["assembly_id"] = assembly_id
        return results


class ChooseFirstAssembly(BaseTransform):
    """Choose the first assembly for test"""

    def transform(self, results: Dict) -> Dict | Tuple[List] | None:
        results["assembly_id"] = list(results["assembly_len_dict"].keys())[0]
        return results


class MPNNLoadCif(BaseTransform):
    """
    MPNN version of loading cif file.
    This transform is used to load the cif file and the corresponding sequence
    """

    def __init__(
        self,
        parser_exclude_res_list=[],
        atoms_to_load="all",
        parse_only_DNA_RNA=False,
        parse_only_proteins=False,  # TODO: merge into one param, like: parse_list=["protein", "ligand"]
        protein_backbone_occ_cutoff=0.8,
        protein_side_chain_occ_cutoff=0.5,
        ligand_occ_cutoff=0.01,
        excluded_elements=[1],
        num_nn=25,
    ):
        """
        Args:
        cache_folder: (cif_base_path, cache_base_path)
        """
        self.parser = CIFParser(skip_res=parser_exclude_res_list)
        if atoms_to_load == "backbone":
            self.atom_list_to_save = ["N", "CA", "C", "O"]
        elif atoms_to_load == "all":
            # fmt: off
            self.atom_list_to_save = [
                'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1',
                'SG', 'CD', 'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD',
                'CE', 'CE1', 'CE2', 'CE3', 'NE', 'NE1', 'NE2', 'OE1', 'OE2',
                'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2', 'CZ3', 'NZ', 'OXT']
            # fmt: on
        else:
            raise ValueError(f"Unknown atoms_to_load: {atoms_to_load}")

        if parse_only_DNA_RNA:
            self.exclude_v_list = ["nonpoly"]
        else:
            self.exclude_v_list = []
        self.parse_only_proteins = parse_only_proteins

        self.protein_backbone_occ_cutoff = protein_backbone_occ_cutoff
        self.protein_side_chain_occ_cutoff = protein_side_chain_occ_cutoff
        self.ligand_occ_cutoff = ligand_occ_cutoff
        self.excluded_elements = excluded_elements
        self.num_nn = num_nn

        self.restype_3to1 = restype_3to1
        self.alphabet = list("ACDEFGHIKLMNPQRSTVWYX")
        self.alphabet_dict = dict(zip(self.alphabet, range(len(self.alphabet))))

        self.bb_idx_list = []
        self.backbone_list = ["N", "CA", "C", "O"]
        for i, atom in enumerate(self.atom_list_to_save):
            if atom in self.backbone_list:
                self.bb_idx_list.append(i)

    def transform(self, results: Dict) -> Dict | Tuple[List] | None:
        """
        Required keys are "cif_path" and "assembly_id"
        """
        cif_path = results["cif_path"]
        assembly_id = results["assembly_id"]

        enable_cache = "cache_path" in results
        if enable_cache:
            cache_path = results["cache_path"].format(assembly_id=assembly_id)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            if os.path.exists(cache_path):
                try:
                    out_dict = torch.load(cache_path)
                except:
                    pass
                else:
                    return out_dict

        chains, asmb, covalei, meta = self.parser.parse(cif_path)

        # ------------------proteins vs not--------------------
        polypeptide_letter_list = []
        ligand_letter_list = []
        for chain_letter, v in chains.items():
            if v.type == "polypeptide(L)":
                polypeptide_letter_list.append(chain_letter)
            else:  # polypeptide(D)
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

            for c, (res_id, atom_tuple) in enumerate(  # c is the index of the residue
                itertools.groupby(atoms, lambda x: x[1])
            ):
                for atom_key in atom_tuple:  # ('A', '4', 'MSE', 'N')
                    atom = chains[poly_chain].atoms[atom_key]
                    if atom_key[2] not in list(self.restype_3to1):
                        if (
                            atom.element not in self.excluded_elements
                            and atom.occ > self.ligand_occ_cutoff
                        ):
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
                    "element": np.stack(extra_ligand_element_list),
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
                "residue_idx": residue_idx[mask_occ],
            }

        ligand_chain_dict = {}
        for letter in ligand_letter_list:
            xyz_list = []
            element_list = []
            type_list = []
            for k, v in chains[letter].atoms.items():
                if v.element not in self.excluded_elements:
                    if v.occ > self.ligand_occ_cutoff:
                        xyz_list.append(np.array(v.xyz))
                        element_list.append(np.array(v.element))
                        res_name = v.name[2]

            if len(xyz_list) > 0:
                ligand_chain_dict[letter] = {
                    "xyz": np.stack(xyz_list),
                    "element": np.stack(element_list),
                    "res_type": type_list,
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

        asmb_ids = list(asmb.keys())
        # only keep asmb_ids that have protein chains
        asmb_ids = [
            asmb_id
            for asmb_id in asmb_ids
            if set([i[0] for i in asmb[asmb_id]]) & set(poly_chain_list)
        ]

        if assembly_id not in asmb:
            raise ValueError(f"assembly_id {assembly_id} not found in the cif file")

        for item in asmb[assembly_id]:
            chain_id = item[0]
            rotation_matrix = item[1][:3, :3]
            translation = item[1][:3, 3]

            if chain_id in poly_chain_list:
                chain_data = poly_chain_dict[chain_id]
                tmp_residue_idx = chain_data["residue_idx"]
                xyz = chain_data["xyz"]
                # xyz = np.einsum("ij,raj->rai", rotation_matrix, xyz) + translation
                xyz = np.dot(xyz, rotation_matrix.T) + translation
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
                    (self.alphabet_dict[AA] if AA in self.alphabet else self.alphabet_dict["X"])
                    for AA in seq_one_letter
                ]

                S_list.append(np.array(seq_int, dtype=np.int32))
                R_idx_list.append(tmp_residue_idx)
                chain_labels_list.append(
                    chain_multi * np.ones_like(tmp_residue_idx, dtype=np.int32)
                )
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

        # if len(Cb_list) == 0:
        #     print(f"No protein found in {cif_path}")
        #     raise ValueError(f"No protein found in {cif_path}")
        # Pick closest neighbors per residue using virtual CB distance to context atoms
        if len(Cb_list) > 0:
            Cb_stack = np.concatenate(Cb_list, axis=0)  # [L, 3]
        else:
            Cb_stack = np.zeros([0, 3], dtype=np.float32)
            Y_list = []
        if len(Y_list) > 0:
            Y_stack = np.concatenate(Y_list, axis=0)  # [M, 3]
            Y_t_stack = np.concatenate(Y_t_list, axis=0)  # [M]

            D = np.sum((Cb_stack[:, None, :] - Y_stack[None, :, :]) ** 2, axis=-1)  # [L, M]
            closest_idx = np.argsort(D, axis=-1)[:, : self.num_nn]
            ligand_xyz_closest = np.take_along_axis(
                np.repeat(Y_stack[None, :, :], Cb_stack.shape[0], 0),
                np.repeat(closest_idx[:, :, None], 3, -1),
                axis=1,
            )
            ligand_element_closest = np.take_along_axis(
                np.repeat(Y_t_stack[None, :], Cb_stack.shape[0], 0), closest_idx, axis=1
            )
        else:
            Y_stack = np.zeros([0, 3], dtype=np.float32)
            ligand_xyz_closest = np.zeros([0, self.num_nn, 3], dtype=np.float32)
            ligand_element_closest = np.zeros([0, self.num_nn], dtype=np.float32)

        if len(X_list) > 0:
            X_stack = np.concatenate(X_list, axis=0)  # [L, num_atoms, 3]
            X_m_stack = np.concatenate(X_m_list, axis=0)  # [L, num_atoms]
            S_stack = np.concatenate(S_list, axis=0)  # [L]
            R_idx_stack = np.concatenate(R_idx_list, axis=0)  # [L]
            chain_labels_stack = np.concatenate(chain_labels_list, axis=0)  # [L]
        else:
            X_stack = np.zeros([0, len(self.atom_list_to_save), 3], dtype=np.float32)
            X_m_stack = np.zeros([0, len(self.atom_list_to_save)], dtype=bool)
            S_stack = np.zeros([0], dtype=np.int32)
            R_idx_stack = np.zeros([0], dtype=np.int32)
            chain_labels_stack = np.zeros([0], dtype=np.int32)

        out_dict = {
            "cif_path": cif_path,
            "assembly_id": assembly_id,
            "Y": ligand_xyz_closest.astype(np.float32),
            "Y_t": ligand_element_closest,
            "Y_coords": Y_stack.astype(np.float32),
            "X": X_stack.astype(np.float32),
            "X_m": X_m_stack.astype(bool),
            "S": S_stack.astype(np.int64),
            "seq": np.array([restype_INTtoSTR[i] for i in S_stack], dtype=str),
            "R_idx": R_idx_stack.astype(np.int64),
            "chain_labels": chain_labels_stack.astype(np.int64),
            "protein_L": protein_L,
            "asmb_ids": asmb_ids,
        }

        # save cache
        if enable_cache:
            torch.save(out_dict, cache_path)

        return out_dict


class MPNNLoadPDB(MPNNLoadCif):
    """Load Designed Protein from PDB file"""

    def transform(self, results: Dict) -> Dict:

        cif_path = results["cif_path"]

        enable_cache = "cache_path" in results
        if enable_cache:
            cache_path = results["cache_path"]
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            if os.path.exists(cache_path):
                try:
                    out_dict = torch.load(cache_path)
                except Exception as e:
                    # some cache files may broken
                    print(f"Error loading cache {cache_path}: {e}")
                else:
                    results.update(out_dict)
                    return results

        # TODO: automatically select cif parser or pdb parser
        output_dict, backbone, other_atoms, CA_icodes, water_atoms = parse_PDB(
            cif_path,
            atom_context_num=self.num_nn,
            parse_all_atoms=True,
        )

        X_m = torch.zeros(output_dict["X"].shape[:2]).bool()
        X_m[output_dict["mask"]] = True
        output_dict["X_m"] = X_m

        X = output_dict["X"]
        Y = output_dict["Y"]
        Y_t = output_dict["Y_t"]
        Y_m = output_dict["Y_m"]

        N = X[:, 0, :]
        CA = X[:, 1, :]
        C = X[:, 2, :]

        b = CA - N
        c = C - CA
        a = torch.cross(b, c, axis=-1)
        CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
        Y, Y_t, Y_m, D_XY = get_nearest_neighbours(
            CB, output_dict["mask"], Y, Y_t, Y_m, self.num_nn
        )
        output_dict.update(
            {
                "cif_path": cif_path,
                "Y": Y,
                "Y_t": Y_t,
                "Y_m": Y_m,
            }
        )

        del output_dict["chain_letters"]
        del output_dict["chain_list"]

        # TODO: mask_XY is not in the training code
        # mask_XY = (D_XY < cutoff_for_score)*mask*Y_m[:,0]
        # output_dict["mask_XY"] = mask_XY[None,]

        if enable_cache:
            torch.save(output_dict, cache_path)

        results.update(output_dict)
        return results


class PackInputs(BaseTransform):
    def __init__(self, keys: Sequence[str]):
        self.keys = keys

    def transform(self, results):
        out_dict = {}
        for key in self.keys:
            out_dict[key] = results[key]
        return out_dict


class RandomChoosePlddtPair(BaseTransform):
    """
    Random choose a pair of designs with plddt difference larger than min_gap
    Args:
        min_gap (float): The minimum gap between two designs
        min_chosen_plddt (float): The minimum plddt of the chosen design
        max_rejected_plddt (float): The maximum plddt of the rejected design
    """

    def __init__(self, min_gap=1.0, min_chosen_plddt=None, max_rejected_plddt=None):
        """
        Args:
            min_gap (float): The minimum gap between two designs, 0 - 100
        """
        self.min_gap = min_gap
        self.min_chosen_plddt = min_chosen_plddt
        self.max_rejected_plddt = max_rejected_plddt

    def transform(self, results: Dict):
        designs = results["designs"]
        designs = sorted(designs, key=lambda x: x["plddt"])
        n = len(designs)
        all_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if designs[j]["plddt"] - designs[i]["plddt"] > self.min_gap:
                    all_pairs.append((i, j))
        if len(all_pairs) == 0:
            raise ValueError(f"No pair found: {results['cif_path']}")

        if self.min_chosen_plddt is not None:
            all_pairs = [
                (i, j) for i, j in all_pairs if designs[j]["plddt"] >= self.min_chosen_plddt
            ]
        if self.max_rejected_plddt is not None:
            all_pairs = [
                (i, j) for i, j in all_pairs if designs[i]["plddt"] <= self.max_rejected_plddt
            ]

        i, j = random.choice(all_pairs)
        results["chosen"] = designs[j]
        results["rejected"] = designs[i]
        return results


class ChooseMaxMinPlddtPair(BaseTransform):
    """Choose the pair of designs with the largest plddt difference"""

    def transform(self, results: Dict):
        designs = results["designs"]
        designs = sorted(designs, key=lambda x: x["plddt"])
        n = len(designs)
        i, j = 0, n - 1
        results["chosen"] = designs[j]
        results["rejected"] = designs[i]
        return results
