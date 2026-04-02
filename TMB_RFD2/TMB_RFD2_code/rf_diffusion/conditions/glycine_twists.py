import torch
import itertools
import rf_diffusion.structure as structure
from structure import get_dssp_string
from structure import get_dssp
import random
import numpy as np

NO_GLYCINE_TWISTS = 0
GLYCINE_TWISTS = 1
MASK_GLYCINE_TWISTS = 2
N_CLASS = 3

def print_E_dssp_assignments(indep):
    """
    Prints DSSP secondary structure assignments for each residue in indep
    and lists all residues assigned as loops ('L') in a single list.
    
    Args:
        indep (Indep): The input structure.
    """
    # Get DSSP assignments
    dssp, pairs = get_dssp(indep, compute_pairs=True)
    dssp_string = get_dssp_string(dssp)

    beta_residues = []
    
    # Iterate through DSSP assignments and collect loop residues
    for idx, ss in enumerate(dssp_string):
        # Identify and collect residues labeled as 'L' (loops)
        if ss == 'E':
            beta_residues.append(idx + 1)  # Convert to 1-based index for printing

    # Print all loop residues as a list
    # print(type(loop_residues))
    # print('haha')
    # print("All beta residues (E):", beta_residues)
    # print('here: ', type(beta_residues))
    return beta_residues


def get_refined_glycine_twists(indep, angle_threshold=0.68):
    """
    Identifies glycine twists in a protein structure based on twist angles 
    using a cosine similarity threshold. This function does not consider 
    proximity to Tyr or Trp residues.

    Args:
        indep (Indep): Protein structure object with `xyz` and `seq` attributes.
        angle_threshold (float): Cosine similarity cutoff to identify glycine twists.
        
    Returns:
        torch.Tensor: Refined twist tensor with `1` for glycine twists, 
                      marking the glycine twist position and its ±2 neighbors.
    """
    # Get strand only residues
    betaonly = print_E_dssp_assignments(indep)

    # Ensure `indep` has the required attributes
    if not hasattr(indep, 'xyz') or not hasattr(indep, 'seq'):
        raise AttributeError("`indep` does not have the required attributes 'xyz' and 'seq'.")
    
    num_residues = indep.xyz.shape[0]
    glycine_twist_tensor = torch.zeros(num_residues, dtype=torch.int)  # Initialize tensor with zeros

    # Iterate over each residue, skipping the first two and last two for safe indexing
    for i in range(2, num_residues - 2):
        notstrand = 0
        # Check if the current residue is glycine (encoded as 7)
        if indep.seq[i].item() == 7:
            # Extract the x, y, z coordinates for vector calculations
            vec1 = indep.xyz[i, 1] - indep.xyz[i - 2, 1]
            vec2 = indep.xyz[i + 2, 1] - indep.xyz[i, 1]
            
            # Ensure vec1 and vec2 are 1-dimensional with length 3
            if vec1.shape != (3,) or vec2.shape != (3,):
                continue  # Skip this residue if shapes are incorrect
            
            # Calculate the cosine angle between the two vectors
            cos_angle = np.dot(vec1.numpy(), vec2.numpy()) / (np.linalg.norm(vec1.numpy()) * np.linalg.norm(vec2.numpy()))
            
            

            # Check if the cosine angle is below the specified threshold
            if cos_angle < angle_threshold:
                # Check if the residues are all within a strand and not at either ends
                for resi in list(range(i-2,i+3)):
                    if resi not in betaonly:
                        notstrand = 1
                        break
                if notstrand == 1:
                    continue
                # Retain the glycine twist and its ±2 neighbors
                glycine_twist_tensor[max(0, i - 2):min(num_residues, i + 3)] = 1

#    print("Final glycine twist tensor:", glycine_twist_tensor)
    return glycine_twist_tensor


def get_glycine_twists_conditioning_inference(indep, feature_conf, feature_inference_conf, **kwargs):
    """
    Inference function for glycine twists conditioning.
    Similar to get_hotspots_antihotspots_conditioning_inference.
    """
    return get_glycine_twists_conditioning(indep, feature_conf, **kwargs)


def get_glycine_twists_conditioning(indep, feature_conf, glycine_twists=None, **kwargs):
    """
    Handles the glycine twists condition for feature loading.

    Args:
        indep: Input data.
        feature_conf: Feature configuration.
        glycine_twists: Condition for glycine twists (tensor indicating glycine twists).
        kwargs: Additional arguments.
    
    Returns:
        torch.Tensor: A tensor marking glycine twists.
    """

    # Retrieve the length of the sequence from `indep`
    length = indep.length()

    
    # Initialize a default condition: return a tensor of zeros if no condition is provided
    if glycine_twists is None:
        glycine_twists = torch.full(indep.length(), MASK_GLYCINE_TWISTS, dtype=int)

    # Ensure the return is of the shape [L, n], where n=1 for 1D features (in this case)
    # beta_bulge_tensor = glycine_twists.view(length, 1)
    glycine_twists_tensor = torch.nn.functional.one_hot(glycine_twists, num_classes=N_CLASS).float()

    # Debugging statement for tensor values
    # print(f"Beta bulge conditioning tensor: {beta_bulge_tensor}")

    return {'t1d': glycine_twists_tensor}


# Define the class that applies the transform
class FindGlycineTwistsTrainingTransform:
    """
    Args:
    p_shown: Probability of showing condition (glycine twists)
    glycine twists_min_mask: Minimum amount of masking
    glycine twists_max_mask: Maximum amount of masking
    """
    def __init__(self, p_shown=1.0, glycine_twists_min_mask=0.1, glycine_twists_max_mask=0.7):  # Force the condition to always apply for testing
        self.p_shown = p_shown
        self.glycine_twists_min_mask = glycine_twists_min_mask
        self.glycine_twists_max_mask = glycine_twists_max_mask

    def __call__(self, indep, conditions_dict, **kwargs):
        if torch.rand(1) > self.p_shown:
            self.glycine_twists_min_mask = 1
            self.glycine_twists_max_mask = 1


        glycine_twists = get_refined_glycine_twists(indep).to(torch.long)


        # mask certain propotion of bulges
        mask_prob = random.uniform(self.glycine_twists_min_mask, self.glycine_twists_max_mask)
        num_to_mask = int(((indep.length())) * mask_prob)
        indices_to_mask = random.sample(range(indep.length()), num_to_mask)

        for i in indices_to_mask:
            glycine_twists[i] = MASK_GLYCINE_TWISTS
        conditions_dict['glycine_twists'] = glycine_twists

        return kwargs | dict(indep=indep, conditions_dict=conditions_dict)



class GlycineTwistsResInferenceTransform:
    def __call__(self, conf, indep, conditions_dict, contig_map, **kwargs):
        """
        Apply glycine twists condition to conditions_dict.

        Uses:
            conf.glycine_twists.res

        Args:
            indep (Indep): Indep object.
            conf (OmegaConf): The configuration object.
            contig_map (ContigMap): The contig map object.
            conditions_dict (dict): The inference conditions.

        Returns:
            dict: Updated conditions dictionary with the glycine twists mask.
        """

        # Handling user-specified glycine twist residues
        num_residues = indep.length()
        glycine_twists = torch.zeros((num_residues,))
        print(conf.glycine_twists.glycine_twists_res)
        if bool(conf.glycine_twists.glycine_twists_res):
            assert isinstance(conf.glycine_twists.glycine_twists_res, str), (
                f'glycine_twists.res must be a string! You passed "{conf.glycine_twists.glycine_twists_res}"'
            )
            print("Using glycine twists.")
            glycine_twists_res = torch.tensor([int(x) for x in conf.glycine_twists.glycine_twists_res.split(",")]) - 1

            # if "is_beta_bulge" in conditions_dict:
            #     conditions_dict["glycine_twists"] |= glycine_twists_mask
            # else:
            #     conditions_dict["glycine_twists"] = glycine_twists_mask

            glycine_twists[glycine_twists_res] = 1
        else:
            print("Not providing glycine twists index, will assume no specified glycine twists.")

        conditions_dict["glycine_twists"] = glycine_twists.to(torch.long)

        return kwargs | dict(
            indep=indep,
            conf=conf,
            contig_map=contig_map,
            conditions_dict=conditions_dict,
        )


