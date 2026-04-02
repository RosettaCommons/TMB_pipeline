"""from {train_folder}/ligand_model_utils_v.py"""

import torch


def train_featurize(batch, mode="backbone", num_nn=4):
    """
    batch: [[dict_item, int(seq_len)], ...]
        dict_item keys: Y, Y_t, X, X_m, S, R_idx, chain_labels, protein_L
    """
    batch = [b for b in batch if type(b[0]) != list]
    B = len(batch)
    if B > 0:
        L_stack = torch.stack([b[1] for b in batch])
        L_max = torch.max(L_stack)
        device = L_max.device
        if mode == "backbone":
            X = torch.zeros([B, L_max, 4, 3], dtype=torch.float32, device=device)
            X_m = torch.zeros([B, L_max, 4], dtype=torch.bool, device=device)
        elif mode == "all":
            X = torch.zeros([B, L_max, 37, 3], dtype=torch.float32, device=device)
            X_m = torch.zeros([B, L_max, 37], dtype=torch.bool, device=device)
        else:
            raise ValueError(f"mode {mode} not recognized")

        Y = torch.zeros([B, L_max, num_nn, 3], dtype=torch.float32, device=device)
        Y_m = torch.zeros([B, L_max, num_nn], dtype=torch.bool, device=device)
        Y_t = torch.zeros([B, L_max, num_nn], dtype=torch.long, device=device)

        mask = torch.zeros([B, L_max], dtype=torch.bool, device=device)
        S = 20 * torch.ones([B, L_max], dtype=torch.long, device=device)
        R_idx = -100 * torch.ones([B, L_max], dtype=torch.long, device=device)
        chain_labels = -1 * torch.ones([B, L_max], dtype=torch.int64, device=device)
        for i, b in enumerate(batch):
            out_dict = b[0]
            X[i, : L_stack[i]] = out_dict["X"][None,]
            X_m[i, : L_stack[i]] = out_dict["X_m"][None,]
            mask[i, : L_stack[i]] = torch.ones_like(out_dict["S"][None,], dtype=torch.bool)
            S[i, : L_stack[i]] = out_dict["S"][None,]
            R_idx[i, : L_stack[i]] = out_dict["R_idx"][None,]
            chain_labels[i, : L_stack[i]] = out_dict["chain_labels"][None,]

            num_atoms = out_dict["Y"].shape[-2]
            Y[i, : L_stack[i], :num_atoms] = out_dict["Y"][None,]
            Y_t[i, : L_stack[i], :num_atoms] = out_dict["Y_t"][None,]
            mask_t_0 = out_dict["Y_t"][None,] != 0
            Y_m[i, : L_stack[i], :num_atoms] = mask_t_0 * torch.ones_like(out_dict["Y_t"][None,], dtype=torch.bool)

        return X, X_m, mask, S, R_idx, chain_labels, Y, Y_t, Y_m

    else:
        raise ValueError("Batch size is 0")


def padding_batch(batch, mode="backbone", num_nn=4):
    """
    Featurizes a batch of protein data for training.

    Args:
    - batch (list of tuples): Each tuple contains a dictionary of data and an integer.
      The dictionary contains keys: 'Y', 'Y_t', 'X', 'X_m', 'S', 'R_idx', 'chain_labels', 'protein_L'.
    - mode (str): The mode of featurization ('backbone' or 'all').
    - num_nn (int): Number of nearest neighbors.

    Returns:
    - Tuple of tensors: (X, X_m, mask, S, R_idx, chain_labels, Y, Y_t, Y_m)
    """
    # Filter out invalid data
    batch = [b for b in batch if not isinstance(b[0], list)]
    B = len(batch)

    if B == 0:
        raise ValueError("Batch size is 0")

    # Determine device from the first element
    device = batch[0][0]['X'].device

    # Get maximum length in batch
    lengths = torch.stack([b[1] for b in batch])
    L_max = torch.max(lengths)

    # Initialize output tensors based on mode
    if mode == "backbone":
        num_atoms = 4
    elif mode == "all":
        num_atoms = 37
    else:
        raise ValueError(f"Mode '{mode}' not recognized")

    # Initialize tensors
    X = torch.zeros((B, L_max, num_atoms, 3), dtype=torch.float32, device=device)
    X_m = torch.zeros((B, L_max, num_atoms), dtype=torch.bool, device=device)
    Y = torch.zeros((B, L_max, num_nn, 3), dtype=torch.float32, device=device)
    Y_m = torch.zeros((B, L_max, num_nn), dtype=torch.bool, device=device)
    Y_t = torch.zeros((B, L_max, num_nn), dtype=torch.long, device=device)
    mask = torch.zeros((B, L_max), dtype=torch.bool, device=device)
    S = 20 * torch.ones((B, L_max), dtype=torch.long, device=device)
    R_idx = -100 * torch.ones((B, L_max), dtype=torch.long, device=device)
    chain_labels = -1 * torch.ones((B, L_max), dtype=torch.int64, device=device)

    # Fill tensors with batch data
    for i, (data, length) in enumerate(batch):
        L = length
        X[i, :L] = data["X"].to(device)
        X_m[i, :L] = data["X_m"].to(device)
        mask[i, :L] = torch.ones(L, dtype=torch.bool, device=device)
        S[i, :L] = data["S"].to(device)
        R_idx[i, :L] = data["R_idx"].to(device)
        chain_labels[i, :L] = data["chain_labels"].to(device)

        num_atoms_in_y = data["Y"].shape[-2]
        Y[i, :L, :num_atoms_in_y] = data["Y"].to(device)
        Y_t[i, :L, :num_atoms_in_y] = data["Y_t"].to(device)
        mask_t_0 = data["Y_t"].to(device) != 0
        Y_m[i, :L, :num_atoms_in_y] = mask_t_0

    return X, X_m, mask, S, R_idx, chain_labels, Y, Y_t, Y_m
