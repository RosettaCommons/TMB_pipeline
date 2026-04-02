import torch
import itertools
import rf_diffusion.structure as structure
import random

NOBULGES = 0
BULGES = 1
MASK_BULGES = 2
N_CLASS = 3


#Function to get beta bulges and return a tensor with extended regions, labeling prolines only in loops
def find_bulges(indep):
    a, b = structure.get_dssp(indep, compute_pairs=True)
    c, d = structure.get_dssp(indep, compute_pairs=False)
    return (a == structure.STRAND) & (c == structure.LOOP)


def get_beta_bulges_conditioning_inference(indep, feature_conf, feature_inference_conf, **kwargs):
    """
    Inference function for beta bulge conditioning.
    Similar to get_hotspots_antihotspots_conditioning_inference.
    """
    return get_beta_bulges_conditioning(indep, feature_conf, **kwargs)


def get_beta_bulges_conditioning(indep, feature_conf, beta_bulges=None, **kwargs):
    """
    Handles the beta bulges condition for feature loading.

    Args:
        indep: Input data (mock or real).
        feature_conf: Feature configuration.
        beta_bulges: Condition for beta bulges (tensor indicating beta bulges).
        kwargs: Additional arguments.
    
    Returns:
        torch.Tensor: A tensor marking beta bulges, shape [L, n] or [L, L, n] depending on feature dimensions.
    """

    # Retrieve the length of the sequence from `indep`
    length = indep.length()
    
    # Initialize a default condition: return a tensor of zeros if no condition is provided
    if beta_bulges is None:
        beta_bulges = torch.full(indep.length(), MASK_BULGES, dtype=int)

    # Ensure the return is of the shape [L, n], where n=1 for 1D features (in this case)
    # beta_bulge_tensor = beta_bulges.view(length, 1)
    beta_bulges_tensor = torch.nn.functional.one_hot(beta_bulges, num_classes=N_CLASS).float()


    # Debugging statement for tensor values
    # print(f"Beta bulge conditioning tensor: {beta_bulge_tensor}")

    return {'t1d': beta_bulges_tensor}

# Define the class that applies the transform
class FindBetaBulgesTrainingTransform:
    """
    Args:
    p_shown: Probability of showing condition (beta bulge)
    bulge_min_mask: Minimum amount of masking
    bulge_max_mask: Maximum amount of masking
    """
    def __init__(self, p_shown=1.0, bulges_min_mask=0.1, bulges_max_mask=0.7):  # Force the condition to always apply for testing
        self.p_shown = p_shown
        self.bulges_min_mask = bulges_min_mask
        self.bulges_max_mask = bulges_max_mask

    def __call__(self, indep, conditions_dict, **kwargs):
        if torch.rand(1) > self.p_shown:
            self.bulges_min_mask = 1
            self.bulges_max_mask = 1

        beta_bulges = find_bulges(indep).to(torch.long)

        # mask certain propotion of bulges
        mask_prob = random.uniform(self.bulges_min_mask, self.bulges_max_mask)
        num_to_mask = int(((indep.length())) * mask_prob)
        indices_to_mask = random.sample(range(indep.length()), num_to_mask)

        for i in indices_to_mask:
            beta_bulges[i] = MASK_BULGES
        conditions_dict['beta_bulges'] = beta_bulges

        return kwargs | dict(indep=indep, conditions_dict=conditions_dict)


class BetaBulgeResInferenceTransform:
    def __call__(self, conf, indep, conditions_dict, contig_map, **kwargs):
        """
        Apply beta bulge condition to conditions_dict.

        Uses:
            conf.beta_bulges.res

        Args:
            indep (Indep): Indep object.
            conf (OmegaConf): The configuration object.
            contig_map (ContigMap): The contig map object.
            conditions_dict (dict): The inference conditions.

        Returns:
            dict: Updated conditions dictionary with the beta bulge mask.
        """

        # Handling user-specified beta bulge residues
        num_residues = indep.length()
        beta_bulges = torch.zeros((num_residues,))
        print(conf.beta_bulges.beta_bulges_res)
        if bool(conf.beta_bulges.beta_bulges_res):
            assert isinstance(conf.beta_bulges.beta_bulges_res, str), (
                f'beta_bulges.res must be a string! You passed "{conf.beta_bulges.beta_bulges_res}"'
            )
            print("Using beta bulges.")
            beta_bulges_res = torch.tensor([int(x) for x in conf.beta_bulges.beta_bulges_res.split(",")]) - 1

            # if "is_beta_bulge" in conditions_dict:
            #     conditions_dict["beta_bulges"] |= beta_bulges_mask
            # else:
            #     conditions_dict["beta_bulges"] = beta_bulges_mask

            beta_bulges_index = torch.tensor(beta_bulges_res) - 1
            beta_bulges[beta_bulges_index] = 1
        else:
            print("Not providing beta bulges index, will assume no specified beta bulges.")

        conditions_dict["beta_bulges"] = beta_bulges.to(torch.long)

        return kwargs | dict(
            indep=indep,
            conf=conf,
            contig_map=contig_map,
            conditions_dict=conditions_dict,
        )



# class BetaBulgeResInferenceTransform:
#     def __call__(self, conf, indep, conditions_dict, contig_map, **kwargs):
#         """
#         Apply beta bulge condition to conditions_dict.

#         Uses:
#             indep (Indep): Indep object.
#             conf (OmegaConf): The configuration object.
#             contig_map (ContigMap): The contig map.
#             conditions_dict (dict): The inference conditions.

#         Returns:
#             dict: Updated conditions dictionary with the beta bulge tensor.
#         """
#         # Assuming your configuration has a flag for beta bulge condition or something similar
#         if hasattr(conf, 'beta_bulges') and bool(conf.beta_bulges):
#             # Generate the beta bulge tensor (or retrieve it if it's precomputed)
#             beta_bulge_tensor = find_bulges(indep)
            
#             # Store or update the 'is_beta_bulge' condition in conditions_dict
#             if 'is_beta_bulge' in conditions_dict:
#                 conditions_dict['is_beta_bulge'] |= beta_bulge_tensor
#             else:
#                 conditions_dict['is_beta_bulge'] = beta_bulge_tensor

#             # Optionally, store other values like a bulge intensity or feature
#             # conditions_dict['beta_bulge_value'] = beta_bulge_tensor * some_value

#         return kwargs | dict(
#             indep=indep,
#             conf=conf,
#             contig_map=contig_map,
#             conditions_dict=conditions_dict,
#         )

# torch.set_printoptions(threshold=torch.inf)

#############################################################################################################################


# # Function to test proline identification in PDB file
# def get_beta_bulges_tensor_from_pdb(pdb_file_path) -> torch.Tensor:
#     """
#     Given a PDB file path, returns a tensor marking positions of proline residues,
#     ensuring the residue number corresponds to the position in the tensor.

#     Args:
#     pdb_file_path (str): Path to the PDB file.

#     Returns:
#     torch.Tensor: A tensor with 1s at positions corresponding to proline residues.
#     """
#     # Initialize variables
#     residues = {}
#     seen_residues = set()  # To avoid duplicate residues
#     max_residue_number = 0  # Track the largest residue number to size the tensor

#     try:
#         # Open the PDB file and read lines
#         with open(pdb_file_path, 'r', encoding='utf-8') as file:
#             for line in file:
#                 if line.startswith("ATOM"):
#                     # Split the line into columns (assuming no chain consideration)
#                     columns = line.split()

#                     # Extract residue number and residue name
#                     residue_number = int(columns[5])  # Convert to integer for correct ordering
#                     residue_name = columns[3]

#                     # Keep track of the maximum residue number for tensor sizing
#                     if residue_number > max_residue_number:
#                         max_residue_number = residue_number

#                     # Only add the residue if it hasn't been added already
#                     if residue_number not in seen_residues:
#                         residues[residue_number] = residue_name
#                         seen_residues.add(residue_number)

#     except UnicodeDecodeError:
#         # Retry with latin1 encoding if utf-8 fails
#         with open(pdb_file_path, 'r', encoding='latin1') as file:
#             for line in file:
#                 if line.startswith("ATOM"):
#                     columns = line.split()
#                     residue_number = int(columns[5])
#                     residue_name = columns[3]
#                     if residue_number not in seen_residues:
#                         residues[residue_number] = residue_name
#                         seen_residues.add(residue_number)

#     # Initialize a tensor with zeros, where the size is the max residue number
#     bulge_tensor = torch.zeros(max_residue_number + 1, dtype=torch.int)

#     # Iterate through the residues and mark prolines (residue name 'PRO') in the correct positions
#     for residue_number, residue_name in residues.items():
#         if residue_name == 'PRO':
#             bulge_tensor[residue_number] = 1

#     # Print the tensor for verification
#     print(f"Bulge tensor for {pdb_file_path}:\n{bulge_tensor}")
    
#     # Print the residue numbers of prolines found
#     proline_indices = (bulge_tensor == 1).nonzero(as_tuple=True)[0]
#     if len(proline_indices) > 0:
#         print("Prolines found at the following residue positions:")
#         for index in proline_indices:
#             print(f"Residue number: {index}")
#     else:
#         print("No proline residues found.")

#     return bulge_tensor

# # Example usage with your PDB file
# pdb_file = "/home/aphilo/good_beta_barrels/TMB12_3.pdb"
# get_beta_bulges_tensor_from_pdb(pdb_file)
