import torch
import sys
import numpy as np
import rf_diffusion.structure as structure
from structure import get_dssp_string
from structure import get_dssp

from run_inference import match_guideposts_and_generate_mappings

class FilterFailed(Exception):
    def __init__(self, i_t, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.i_t = i_t

    pass


def init_filters(conf):

    filters = []
    if not hasattr(conf, 'filters'):
        return filters
    if not hasattr(conf.filters, 'names'):
        return filters

    assert hasattr(conf.filters, 'names')
    assert hasattr(conf.filters, 'configs')

    thismodule = sys.modules[__name__]

    for name in conf.filters.names:
        if ':' in name:
            filter_name, class_name = name.split(':')
        else:
            filter_name = class_name = name
        filters.append(getattr(thismodule, class_name)(name=filter_name, **getattr(conf.filters.configs, filter_name)))

    return filters

def do_filtering(filters, indep, t, i_t, px0, scores, **kwargs):

    if len(filters) == 0:
        return

    kwargs['indep'] = indep
    kwargs['t'] = t

    px0_indep = indep.clone()
    px0_indep.xyz = px0
    kwargs['px0_indep'] = px0_indep

    passing = True
    for filt in filters:
        local_passing, local_scores = filt.do_filter(**kwargs)
        passing = passing and local_passing
        scores.update(local_scores)

    if not passing:
        raise FilterFailed(i_t)
    
    
# def print_E_dssp_assignments(indep):
#     """
#     Prints DSSP secondary structure assignments for each residue in indep
#     and lists all residues assigned as loops ('L') in a single list.
    
#     Args:
#         indep (Indep): The input structure.
#     """
#     # Get DSSP assignments
#     dssp, pairs = get_dssp(indep, compute_pairs=False)
#     dssp_string = get_dssp_string(dssp)
    
    

#     beta_residues = []
    
#     # Iterate through DSSP assignments and collect loop residues
#     for idx, ss in enumerate(dssp_string):
#         # Identify and collect residues labeled as 'L' (loops)
#         if ss == 'E':
#             beta_residues.append(idx + 1)  # Convert to 1-based index for printing

#     # Print all loop residues as a list
#     # print(type(loop_residues))
#     # print('haha')
#     # print("All beta residues (E):", beta_residues)
#     # print('here: ', type(beta_residues))
#     return beta_residues
    
    
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
        #betaonly = print_E_dssp_assignments(indep)
        

        # Ensure `indep` has the required attributes
        if not hasattr(indep, 'xyz') or not hasattr(indep, 'seq'):
            raise AttributeError("`indep` does not have the required attributes 'xyz' and 'seq'.")
        
        num_residues = indep.xyz.shape[0]
        glycine_twist_tensor = torch.zeros(num_residues, dtype=torch.int)  # Initialize tensor with zeros

        # Iterate over each residue, skipping the first two and last two for safe indexing
        for i in range(2, num_residues - 2):
            # Check if the current residue is glycine (encoded as 7)
#            if indep.seq[i].item() == 7:
            # Extract the x, y, z coordinates for vector calculations
            vec1 = indep.xyz[i, 1] - indep.xyz[i - 2, 1]
            vec2 = indep.xyz[i + 2, 1] - indep.xyz[i, 1]
            
            # Ensure vec1 and vec2 are 1-dimensional with length 3
            if vec1.shape != (3,) or vec2.shape != (3,):
                continue  # Skip this residue if shapes are incorrect
            
            # Calculate the cosine angle between the two vectors
            cos_angle = np.dot(vec1.numpy(), vec2.numpy()) / (np.linalg.norm(vec1.numpy()) * np.linalg.norm(vec2.numpy()))
            
            # Check if the cosine angle is below the specified threshold
            if cos_angle < angle_threshold and cos_angle > 0.6:
                #notstrand = 0
                # Check if the residues are all within a strand and not at either ends
                # for resi in list(range(i-2,i+3)):
                #     if resi not in betaonly:
                #         notstrand += 1
                        
                # if notstrand >= 2:
                #     continue
                # Retain the glycine twist and its ±2 neighbors
                glycine_twist_tensor[max(0, i - 2):min(num_residues, i + 3)] = 1

    #    print("Final glycine twist tensor:", glycine_twist_tensor)
        return glycine_twist_tensor



class FilterBase:

    def __init__(self, t=None, suffix=None, prefix=None, verbose=False, name=None):
        self.suffix = suffix or ''
        self.prefix = prefix or ''
        self.verbose = verbose
        if name is None:
            name = self.__class__.__name__
        self.name = name

        self.active_t = set()
        try:
            t = int(t)
            self.active_t.add(t)
        except ValueError as _:
            if t is not None:
                for inner_t in t.split(','):
                    inner_t = int(inner_t)
                    self.active_t.add(inner_t)

    def do_filter(self, t, **kwargs):

        assert hasattr(self, 'active_t'), 'You forgot to call super().__init__(**kwargs)'

        # int() to drop tensor
        if int(t) not in self.active_t:
            return True, {}

        passing, scores = self.inner_do_filter(t=t, **kwargs)

        if self.verbose or not passing:
            print(f'Filter: {self.name} at t:{t} {"passes" if passing else "fails"} with: {scores}')

        final_scores = {}
        for key, value in scores.items():
            final_scores[f'{self.prefix}t-{t}_{key}{self.suffix}'] = value

        return passing, final_scores


    def inner_do_filter(self, **kwargs):
        assert False, 'FilterBase.inner_do_filter() was called! You should overwrite this!'



class ChainBreak(FilterBase):

    def __init__(self, C_N_dist_limit=None, monitor_chains='0', use_px0=True, **kwargs):
        super().__init__(**kwargs)

        self.C_N_dist_limit = C_N_dist_limit if C_N_dist_limit is not None else 9e9
        self.use_px0 = use_px0

        if monitor_chains is None:
            self.monitor_chains = None
        else:
            monitor = set()
            try:
                monitor.add(int(monitor_chains))
            except ValueError as _:
                for inner_chain in monitor_chains.split(','):
                    monitor.add(int(inner_chain))

            self.monitor_chains = monitor

    def inner_do_filter(self, indep, px0_indep, **kwargs):

        if self.use_px0:
            indep = px0_indep

        scores = {}
        scores['max_chain_gap'] = 0
        for ichain, chain_mask in enumerate(indep.chain_masks()):
            chain_mask = torch.tensor(chain_mask)
            chain_mask[indep.is_gp] = False

            if self.monitor_chains is not None:
                if ichain not in self.monitor_chains:
                    continue

            # If there are any small molecules in this chain, just clear out everything past the first sm because
            #  idk what's going on with the connectivity at that point
            mask_is_sm = indep.is_sm & chain_mask
            wh_sm = torch.where(mask_is_sm)[0]
            if len(wh_sm) > 0:
                first_sm = wh_sm[0]
                chain_mask[first_sm:] = False

            if chain_mask.sum() <= 1:
                continue

            local_Ns = indep.xyz[chain_mask,0]
            local_Cs = indep.xyz[chain_mask,2]

            C_from_N = local_Cs[:-1] - local_Ns[1:]
            C_N_dists = torch.linalg.norm( C_from_N, axis=-1)
            max_gap = C_N_dists.max()

            scores['max_chain_gap'] = max(scores['max_chain_gap'], max_gap)

        return scores['max_chain_gap'] <= self.C_N_dist_limit, scores




class GPSatisfaction(FilterBase):

    def __init__(self, use_px0=True, gp_max_error_cut=None, gp_rmsd_cut=None, **kwargs):
        super().__init__(**kwargs)

        self.use_px0 = use_px0
        self.gp_max_error_cut = gp_max_error_cut if gp_max_error_cut is not None else 9e9
        self.gp_rmsd_cut = gp_rmsd_cut if gp_rmsd_cut is not None else 9e9


    def inner_do_filter(self, indep, px0_indep, contig_map, is_diffused, **kwargs):


        scores = {}
        scores['gp_max_error'] = 0
        scores['gp_rmsd'] = 0

        xyz = px0_indep.xyz if self.use_px0 else indep.xyz
        match_idx, gp_idx, _ = match_guideposts_and_generate_mappings(indep, is_diffused, contig_map, xyz)

        match_CAs = xyz[match_idx,1]
        gp_CAs = xyz[gp_idx,1]

        if len(match_CAs) == 0:
            return True, scores

        distances = torch.linalg.norm( match_CAs - gp_CAs, axis=-1 )

        scores['gp_max_error'] = distances.max()
        scores['gp_rmsd'] = torch.sqrt( torch.mean( torch.square( distances) ) )

        passing = True
        passing = passing and scores['gp_max_error'] < self.gp_max_error_cut
        passing = passing and scores['gp_rmsd'] < self.gp_rmsd_cut

        return passing, scores
    
    
class GlycineKinkCheck(FilterBase):
    def __init__(self, use_px0=True, kink_num_cutoff=None, **kwargs):
        super().__init__(**kwargs)
        
        self.use_px0 = use_px0
        self.kink_num_cutoff = kink_num_cutoff if kink_num_cutoff is not None else 0
    


    def inner_do_filter(self, indep, px0_indep, **kwargs):
        scores = {}
        scores['glycine_kink'] = 0

        # Check for glycine kinks by finding all consecutive 1s in the glycine twist tensor
        if self.use_px0:
            glycine_twist_tensor = get_refined_glycine_twists(px0_indep)
        else:
            glycine_twist_tensor = get_refined_glycine_twists(indep)
        # Find the indices of all glycine kinks
        glycine_kinks = np.where(glycine_twist_tensor.numpy() == 1)[0]
        # find all sets of consecutive 1s
        consecutive_kinks = []
        current_set = []
        for i in range(len(glycine_kinks)):
            if i == 0 or glycine_kinks[i] == glycine_kinks[i-1] + 1:
                current_set.append(glycine_kinks[i])
            else:
                if len(current_set) == 5:
                    consecutive_kinks.append(current_set)
                current_set = [glycine_kinks[i]]
        if len(current_set) == 5:
            consecutive_kinks.append(current_set)
            
            
        print('consecutive_kinks: ', consecutive_kinks)
            
        # Count the number of sets of consecutive 1s
        scores['glycine_kink'] = len(consecutive_kinks)

        return scores['glycine_kink'] >= self.kink_num_cutoff, scores










