import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import utils, residue_constants, quat_affine, all_atom, r3
#from . import all_atom_multimer as all_atom
from .common_modules import *
from typing import Dict, Optional
import functools, functorch
import scipy.special

from .geometry.vector import Vec3Array, euclidean_distance

def squared_difference(x, y):
    return torch.square(x - y)

####################################
### masked_msa_head
####################################

class MaskedMsaHead(nn.Module):
    """Head to predict MSA at the masked locations.
    
    The MaskedMsaHead employs a BERT-style objective to reconstruct a masked
    version of the full MSA, based on a linear projection of
    the MSA representation.
    Jumper et al. (2021) Suppl. Sec. 1.9.9 "Masked MSA prediction"
    """
    
    def __init__(self, msa_dim=256, num_output=23, zero_init=True):
        super().__init__()
        self.msa_dim = msa_dim
        self.num_output = num_output
        self.zero_init = zero_init
        
        self.logits = Linear(msa_dim, num_output, initializer='zeros' if zero_init else 'linear')
    
    def forward(self, representations, batch=None):
        """Builds MaskedMsaHead module.
        Arguments
        -------------
        msa_act: [N_seq, N_res, msa_dim]
        batch: dict, unused
        
        Return
        -------------
        logits: [N_seq, N_res, num_output]
        """
        assert representations['msa'].ndim == 3
        assert representations['msa'].shape[2] == self.msa_dim
        
        logits = self.logits(representations['msa'])
        return dict(logits=logits)


####################################
### distogram_head
####################################
    
class DistogramHead(nn.Module):
    """Head to predict a distogram.
    
    Jumper et al. (2021) Suppl. Sec. 1.9.8 "Distogram prediction"
    """
    def __init__(self, pair_dim=128, first_break=2.3125, last_break=21.6875, num_bins=64, zero_init=True):
        super().__init__()
        
        self.pair_dim = pair_dim
        self.first_break = first_break
        self.last_break = last_break
        self.num_bins = num_bins
        
        self.half_logits = Linear(pair_dim, num_bins, initializer='zeros' if zero_init else 'linear')
    
    def forward(self, representations, batch):
        """Builds DistogramHead module.
        
        representations: dict
            -- pair: [N_res, N_res, c_z]
        
        batch: unused
        
        Returns:
            Dict contains:
             -- logits: [N_res, N_res, N_bins]
             -- bin_edges: [N_bins - 1,]
        """
        assert representations['pair'].ndim == 3
        assert representations['pair'].shape[2] == self.pair_dim
        
        half_logits = self.half_logits( representations['pair'] )  # [N_res, N_res, num_bins]
        logits = half_logits + torch.swapaxes(half_logits, -2, -3) # [N_res, N_res, num_bins]
        breaks = torch.linspace(self.first_break, self.last_break, self.num_bins - 1).to(logits)
        
        return dict(logits=logits, bin_edges=breaks)



####################################
### predicted_lddt_head
####################################

class PredictedLDDTHead(nn.Module):
    """Head to predict the per-residue LDDT to be used as a confidence measure.

    Jumper et al. (2021) Suppl. Sec. 1.9.6 "Model confidence prediction (pLDDT)"
    Jumper et al. (2021) Suppl. Alg. 29 "predictPerResidueLDDT_Ca"
    """

    def __init__(self, single_dim=384, num_channels=128, num_bins=50, zero_init=True, min_resolution=0.1, max_resolution=3.0, filter_by_resolution=False):
        super().__init__()

        self.single_dim = single_dim
        self.num_channels = num_channels
        self.num_bins = num_bins
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.filter_by_resolution = filter_by_resolution

        self.input_layer_norm = LayerNorm(single_dim)
        self.act_0 = Linear(single_dim, num_channels, initializer='relu')
        self.act_1 = Linear(num_channels, num_channels, initializer='relu')
        self.logits = Linear(num_channels, num_bins, initializer='zeros' if zero_init else 'linear')

    def forward(self, representations, batch):
        """Builds ExperimentallyResolvedHead module.
        
        Arguments
        ------------
        representations: dict
            -- structure_module
               -- single: [N_res, c_s]
          batch: Batch, unused.

        Returns
        -----------
          dict containing:
            -- logits: [N_res, N_bins]
        """
        act = representations['structure_module']

        act = self.input_layer_norm(act)
        act = self.act_0(act)
        act = torch.relu(act)
        act = self.act_1(act)
        act = torch.relu(act)

        logits = self.logits(act) # [B, N_res, num_bins]

        return dict(logits=logits)


def lddt(predicted_points, true_points, true_points_mask, cutoff=15., per_residue=False):
    """Measure (approximate) lDDT for a batch of coordinates.

    lDDT reference:
    Mariani, V., Biasini, M., Barbato, A. & Schwede, T. lDDT: A local
    superposition-free score for comparing protein structures and models using
    distance difference tests. Bioinformatics 29, 2722–2728 (2013).

    lDDT is a measure of the difference between the true distance matrix and the
    distance matrix of the predicted points.  The difference is computed only on
    points closer than cutoff *in the true structure*.

    This function does not compute the exact lDDT value that the original paper
    describes because it does not include terms for physical feasibility
    (e.g. bond length violations). Therefore this is only an approximate
    lDDT score.

    Arguments
    -------------
    predicted_points: [batch, N_res, 3]
    true_points: [batch, N_res, 3] 
    true_points_mask: [batch, N_res, 1] binary-valued float array. This mask
      should be 1 for points that exist in the true points.
    cutoff: Maximum distance for a pair of points to be included
    per_residue: If true, return score for each residue.  Note that the overall
      lDDT is not exactly the mean of the per_residue lDDT's because some
      residues have more contacts than others.

    Returns
    -------------
    An (approximate, see above) lDDT score in the range 0-1.
    """

    assert predicted_points.ndim == 3
    assert true_points_mask.ndim == 3
    assert predicted_points.shape[-1] == 3
    assert true_points_mask.shape[-1] == 1
    
    # Compute true and predicted distance matrices.
    dmat_true = torch.sqrt(1e-10 + torch.sum((true_points[:, :, None] - true_points[:, None, :])**2, dim=-1))

    dmat_predicted = torch.sqrt(1e-10 + torch.sum((predicted_points[:, :, None] - predicted_points[:, None, :])**2, dim=-1))

    dists_to_score = (
      (dmat_true < cutoff).to(predicted_points) * 
      true_points_mask * 
      torch.permute(true_points_mask, [0, 2, 1]) * 
      (1. - torch.eye(dmat_true.shape[1])).to(predicted_points)  # Exclude self-interaction.
    )

    # Shift unscored distances to be far away.
    dist_l1 = torch.abs(dmat_true - dmat_predicted)

    # True lDDT uses a number of fixed bins.
    # We ignore the physical plausibility correction to lDDT, though.
    score = 0.25 * ((dist_l1 < 0.5).to(predicted_points) +
                  (dist_l1 < 1.0).to(predicted_points) +
                  (dist_l1 < 2.0).to(predicted_points) +
                  (dist_l1 < 4.0).to(predicted_points))

    # Normalize over the appropriate axes.
    reduce_axes = (-1,) if per_residue else (-2, -1)
    norm = 1. / (1e-10 + torch.sum(dists_to_score, dim=reduce_axes))
    score = norm * (1e-10 + torch.sum(dists_to_score * score, dim=reduce_axes))

    return score

def compute_plddt(logits):
    """Computes per-residue pLDDT from logits.

    Args:
    logits: [num_res, num_bins] output from the PredictedLDDTHead.

    Returns:
    plddt: [num_res] per-residue pLDDT.
    """
    import scipy.special
    
    num_bins = logits.shape[-1]
    bin_width = 1.0 / num_bins
    bin_centers = np.arange(start=0.5 * bin_width, stop=1.0, step=bin_width)
    probs = scipy.special.softmax(logits, axis=-1)
    predicted_lddt_ca = np.sum(probs * bin_centers[None, :], axis=-1)
    return predicted_lddt_ca * 100

def compute_predicted_aligned_error(logits, breaks):
    """Computes aligned confidence metrics from logits.

    Args:
    logits: [num_res, num_res, num_bins] the logits output from
      PredictedAlignedErrorHead.
    breaks: [num_bins - 1] the error bin edges.

    Returns:
    aligned_confidence_probs: [num_res, num_res, num_bins] the predicted
      aligned error probabilities over bins for each residue pair.
    predicted_aligned_error: [num_res, num_res] the expected aligned distance
      error for each pair of residues.
    max_predicted_aligned_error: The maximum predicted error possible.
    """
    import scipy.special
    
    aligned_confidence_probs = scipy.special.softmax( logits, axis=-1)
    predicted_aligned_error, max_predicted_aligned_error = (
      _calculate_expected_aligned_error(breaks, aligned_confidence_probs))
    return {
      'aligned_confidence_probs': aligned_confidence_probs,
      'predicted_aligned_error': predicted_aligned_error,
      'max_predicted_aligned_error': max_predicted_aligned_error,
    }

def _calculate_expected_aligned_error(alignment_confidence_breaks, aligned_distance_error_probs):
    """Calculates expected aligned distance errors for every pair of residues.

    Args:
    alignment_confidence_breaks: [num_bins - 1] the error bin edges.
    aligned_distance_error_probs: [num_res, num_res, num_bins] the predicted
      probs for each error bin, for each pair of residues.

    Returns:
    predicted_aligned_error: [num_res, num_res] the expected aligned distance
      error for each pair of residues.
    max_predicted_aligned_error: The maximum predicted error possible.
    """
    bin_centers = _calculate_bin_centers(alignment_confidence_breaks)

    # Tuple of expected aligned distance error and max possible error.
    return (np.sum(aligned_distance_error_probs * bin_centers, axis=-1), np.asarray(bin_centers[-1]))

def _calculate_bin_centers(breaks):
    """Gets the bin centers from the bin edges.

    Args:
    breaks: [num_bins - 1] the error bin edges.

    Returns:
    bin_centers: [num_bins] the error bin centers.
    """
    step = (breaks[1] - breaks[0])

    # Add half-step to get the center
    bin_centers = breaks + step / 2
    # Add a catch-all bin at the end.
    bin_centers = np.concatenate([bin_centers, [bin_centers[-1] + step]], axis=0)
    return bin_centers

####################################
### predicted_aligned_error_head
####################################

class PredictedAlignedErrorHead(nn.Module):
    """Head to predict the distance errors in the backbone alignment frames.

    Can be used to compute predicted TM-Score.
    Jumper et al. (2021) Suppl. Sec. 1.9.7 "TM-score prediction"
    """

    def __init__(self, pair_dim=128, max_error_bin=31, num_bins=64, num_channels=128, 
                       min_resolution=0.1, max_resolution=3.0, filter_by_resolution=False):
        super().__init__()

        self.pair_dim = pair_dim
        self.max_error_bin = max_error_bin
        self.num_bins = num_bins
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.filter_by_resolution = filter_by_resolution

        self.logits = Linear(pair_dim, num_bins)

    def forward(self, representations, batch):
        """Builds PredictedAlignedErrorHead module.

        Arguments
        ----------
          representations: dict
            -- pair: [N_res, N_res, c_z]
          batch: unused.

        Returns
        ---------
          dict containing:
            -- logits: [N_res, N_res, N_bins].
            -- bin_breaks: [N_bins - 1]
        """

        act = representations['pair']
        
        logits = self.logits(act) # [N_res, N_res, num_bins]
        breaks = torch.linspace(0., self.max_error_bin, self.num_bins - 1) # [num_bins]
        return dict(logits=logits, breaks=breaks)


def predicted_tm_score(logits, breaks, residue_weights=None):
    """Computes predicted TM alignment score.

    Args:
    logits: [num_res, num_res, num_bins] the logits output from
      PredictedAlignedErrorHead.
    breaks: [num_bins] the error bins.
    residue_weights: [num_res] the per residue weights to use for the
      expectation.

    Returns:
    ptm_score: the predicted TM alignment score.
    """

    # residue_weights has to be in [0, 1], but can be floating-point, i.e. the
    # exp. resolved head's probability.
    if residue_weights is None:
        residue_weights = np.ones(logits.shape[0])

    bin_centers = _calculate_bin_centers(breaks)

    num_res = np.sum(residue_weights)
    # Clip num_res to avoid negative/undefined d0.
    clipped_num_res = max(num_res, 19)

    # Compute d_0(num_res) as defined by TM-score, eqn. (5) in
    # http://zhanglab.ccmb.med.umich.edu/papers/2004_3.pdf
    # Yang & Skolnick "Scoring function for automated
    # assessment of protein structure template quality" 2004
    d0 = 1.24 * (clipped_num_res - 15) ** (1./3) - 1.8

    # Convert logits to probs
    probs = scipy.special.softmax(logits, axis=-1)

    # TM-Score term for every bin
    tm_per_bin = 1. / (1 + np.square(bin_centers) / np.square(d0))
    # E_distances tm(distance)
    predicted_tm_term = np.sum(probs * tm_per_bin, axis=-1)

    normed_residue_mask = residue_weights / (1e-8 + residue_weights.sum())
    per_alignment = np.sum(predicted_tm_term * normed_residue_mask, axis=-1)
    return np.asarray(per_alignment[(per_alignment * residue_weights).argmax()])
    
####################################
### experimentally_resolved_head
####################################

class ExperimentallyResolvedHead(nn.Module):
    """Predicts if an atom is experimentally resolved in a high-res structure.

    Only trained on high-resolution X-ray crystals & cryo-EM.
    Jumper et al. (2021) Suppl. Sec. 1.9.10 '"Experimentally resolved" prediction'
    """

    def __init__(self, single_dim=384, min_resolution=0.1, max_resolution=3.0, filter_by_resolution=False, zero_init=True):
        
        super().__init__()
        
        self.single_dim = single_dim
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.filter_by_resolution = filter_by_resolution

        self.logits = Linear(single_dim, 37, initializer='zeros' if zero_init else 'linear')

    def forward(self, representations, batch):
        """Builds ExperimentallyResolvedHead module.

        Arguments
        ------------
          representations: dict
            -- single: [N_res, c_s]
          batch: unused

        Returns
        ------------
          dict containing:
            -- logits: [N_res, 37]
                log probability that an atom is resolved in atom37 representation,
                can be converted to probability by applying sigmoid.
        """
        logits = self.logits(representations['single'])
        return dict(logits=logits)




def get_rc_tensor(rc_np, aatype):
    return torch.tensor(rc_np, device=aatype.device)[aatype]

def masked_mean(mask, value, dim, eps=1e-4):
    mask = mask.expand(*value.shape)
    return torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))





