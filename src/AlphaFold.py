import os, sys
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import utils, Evoformer, folding, aux_heads


class AlphaFoldIteration(nn.Module):
    def __init__(self, target_feat_dim=22, 
                 msa_feat_dim=49,
                 msa_channel=256, 
                 evoformer_num_block=48,
                 extra_msa_channel=64, 
                 extra_msa_stack_num_block=4, 
                 pair_channel=128, 
                 seq_channel=384, 
                 max_relative_feature=32,
                 gating=True, 
                 zero_init=True, 
                 recycle_features=True, 
                 recycle_pos=True, 
                 enable_template=False, 
                 embed_torsion_angles=False,     # no template atom positions
                 filter_by_resolution=False,
                 # Heads
                 use_msa_head=True,
                 use_structure_module_head=True,
                 use_distogram_head=True,
                 use_predicted_lddt_head=True,
                 use_predicted_aligned_error_head=False,
                 use_experimentally_resolved_head=True):
        super().__init__()
        
        self.use_msa_head = use_msa_head
        self.use_structure_module_head = use_structure_module_head
        self.use_distogram_head = use_distogram_head
        self.use_predicted_lddt_head = use_predicted_lddt_head
        self.use_predicted_aligned_error_head = use_predicted_aligned_error_head
        self.use_experimentally_resolved_head = use_experimentally_resolved_head
        
        self.evoformer = Evoformer.EmbeddingsAndEvoformer(
                target_feat_dim=target_feat_dim,
                msa_feat_dim=msa_feat_dim,
                msa_channel=msa_channel,
                evoformer_num_block=evoformer_num_block,
                extra_msa_channel=extra_msa_channel,
                extra_msa_stack_num_block=extra_msa_stack_num_block,
                pair_channel=pair_channel,
                seq_channel=seq_channel,
                max_relative_feature=max_relative_feature,
                gating=gating,
                zero_init=zero_init,
                recycle_features=recycle_features,
                recycle_pos=recycle_pos,
                enable_template=enable_template,
                embed_torsion_angles=embed_torsion_angles)
        
        if self.use_structure_module_head:
            self.structure_module = folding.StructureModule(
                    single_dim=seq_channel,
                    pair_dim=pair_channel,
                    num_layer=8,
                    sc_num_channels=128,
                    compute_loss=False,
                    zero_init=zero_init,
                    deterministic=False)
        
        if self.use_msa_head:
            self.masked_msa_head = aux_heads.MaskedMsaHead(
                    msa_dim=msa_channel, 
                    num_output=23, 
                    zero_init=zero_init)
        
        if self.use_distogram_head:
            self.distogram_head = aux_heads.DistogramHead(
                    pair_dim=pair_channel, 
                    first_break=2.3125, 
                    last_break=21.6875, 
                    num_bins=64, 
                    zero_init=zero_init)
        
        if self.use_predicted_lddt_head:
            self.predicted_lddt_head = aux_heads.PredictedLDDTHead(
                    single_dim=seq_channel, 
                    num_channels=128, 
                    num_bins=50, 
                    zero_init=zero_init, 
                    min_resolution=0.1, 
                    max_resolution=3.0, 
                    filter_by_resolution=filter_by_resolution)
        
        if self.use_predicted_aligned_error_head:
            self.predicted_aligned_error_head = aux_heads.PredictedAlignedErrorHead(
                    pair_dim=pair_channel, 
                    max_error_bin=31, 
                    num_bins=64, 
                    num_channels=128, 
                    min_resolution=0.1, 
                    max_resolution=3.0, 
                    filter_by_resolution=filter_by_resolution)
        
        if self.use_experimentally_resolved_head:
            self.experimentally_resolved_head = aux_heads.ExperimentallyResolvedHead(
                    single_dim=seq_channel, 
                    min_resolution=0.1, 
                    max_resolution=3.0, 
                    filter_by_resolution=filter_by_resolution, 
                    zero_init=zero_init)
    
    def forward(self, ensembled_batch, non_ensembled_batch, ensemble_representations=True, local_ipa=False, recycle_id=None, use_slice=0):
        """
        ensembled_batch: Normal batch
            -- seq_length: [N_ens]
            -- target_feat: [N_ens, N_res, 22]
            -- msa_feat: [N_ens,N_seq, N_res, 49]
            -- seq_mask: [N_ens,N_res]
            -- residue_index: [N_ens,N_res], must be torch.long
            -- msa_mask: [N_ens,N_seq, N_res]
            
            -- extra_msa: [N_ens, N_extra_seq, N_res], must be torch.long
            -- extra_msa_mask: [N_ens, N_extra_seq, N_res]
            -- extra_has_deletion: [N_ens, N_extra_seq, N_res]
            -- extra_deletion_value: [N_ens, N_extra_seq, N_res]
        
            -- seq_mask: [N_ens, N_res]
            -- aatype: [N_ens, N_res]
            -- atom14_atom_exists: [N_ens, N_res, 14]
            -- atom37_atom_exists: [N_ens, N_res, 37]
            -- residx_atom37_to_atom14: [N_ens, N_res, 37]
        
            For prediction with template
            -- template_mask -- [N_ens, N_templ]
            -- template_aatype -- [N_ens, N_templ, N_res]
            -- template_pseudo_beta_mask -- [N_ens, N_templ, N_res]
            -- template_pseudo_beta -- [N_ens, N_templ, N_res, 3]
            -- template_all_atom_positions -- [N_ens, N_templ, N_res, 37, 3]
            -- template_all_atom_masks -- [N_ens, N_templ, N_res, 37]
        
        non_ensembled_batch: dict
            -- prev_pos: [N_res, 37, 3]
            -- prev_msa_first_row: [N_res, msa_channel]
            -- prev_pair: [N_res, pair_channel]
        """

        num_ensemble = ensembled_batch['seq_length'].shape[0]

        def slice_batch(i):
            b = {k:v[i] for k, v in ensembled_batch.items()}
            b.update(non_ensembled_batch)
            return b

        batch0 = slice_batch(use_slice)
        representations = self.evoformer(batch0)

        msa_representation = representations['msa']
        del representations['msa']
        
        if ensemble_representations:
            for i in range(1, num_ensemble):
                feats = slice_batch(i)
                representations_update = self.evoformer(feats)
                
                for key in representations:
                    representations[key] = representations_update[key] + representations[key]
        
            for k in representations:
                representations[k] = representations[k] / num_ensemble
        
        representations['msa'] = msa_representation
        batch = batch0
        
        ret = {}
        ret['representations'] = representations
        
        if self.use_structure_module_head:
            ret['structure_module'] = self.structure_module(representations, batch, local_ipa=local_ipa, recycle_id=recycle_id)
        if self.use_msa_head:
            ret['masked_msa'] = self.masked_msa_head(representations, batch)
        if self.use_distogram_head:
            ret['distogram'] = self.distogram_head(representations, batch)
        if self.use_predicted_lddt_head:
            ret['predicted_lddt'] = self.predicted_lddt_head(representations, batch)
        if self.use_predicted_aligned_error_head:
            ret['predicted_aligned_error'] = self.predicted_aligned_error_head(representations, batch)
        if self.use_experimentally_resolved_head:
            ret['experimentally_resolved'] = self.experimentally_resolved_head(representations, batch)
        
        return ret


def predict(input_batch, alphafold_model, num_recycle=3):
    """
    Predict the protein structure with AlphaFoldIteration model
    
    input_batch: dict
        -- seq_length: [N_ens]
        -- target_feat: [N_ens, N_res, 22]
        -- msa_feat: [N_ens,N_seq, N_res, 49]
        -- seq_mask: [N_ens,N_res]
        -- residue_index: [N_ens,N_res], must be torch.long
        -- msa_mask: [N_ens,N_seq, N_res]

        -- extra_msa: [N_ens, N_extra_seq, N_res], must be torch.long
        -- extra_msa_mask: [N_ens, N_extra_seq, N_res]
        -- extra_has_deletion: [N_ens, N_extra_seq, N_res]
        -- extra_deletion_value: [N_ens, N_extra_seq, N_res]

        -- seq_mask: [N_ens, N_res]
        -- aatype: [N_ens, N_res]
        -- atom14_atom_exists: [N_ens, N_res, 14]
        -- atom37_atom_exists: [N_ens, N_res, 37]
        -- residx_atom37_to_atom14: [N_ens, N_res, 37]

        For prediction with template
        -- template_mask -- [N_ens, N_templ]
        -- template_aatype -- [N_ens, N_templ, N_res]
        -- template_pseudo_beta_mask -- [N_ens, N_templ, N_res]
        -- template_pseudo_beta -- [N_ens, N_templ, N_res, 3]
        -- template_all_atom_positions -- [N_ens, N_templ, N_res, 37, 3]
        -- template_all_atom_masks -- [N_ens, N_templ, N_res, 37]
        
    input_batch: AlphaFoldIteration
    num_recycle: int 
        Number of recycles
    """
    assert isinstance(alphafold_model, AlphaFoldIteration)
    
    ############# Make fake prev tensor #############
    N_ens, N_res = input_batch['aatype'].shape[:2]
    non_ensembled_batch = {}
    if num_recycle > 0:
        non_ensembled_batch = {
            'prev_pos': torch.zeros([N_res, 37, 3]).to(input_batch['msa_feat']), 
            'prev_msa_first_row': torch.zeros([N_res, 256]).to(input_batch['msa_feat']),
            'prev_pair': torch.zeros([N_res, N_res, 128]).to(input_batch['msa_feat'])
        }
        
    if N_ens == 1:
        ensemble_representations = False
    else:
        ensemble_representations = True

    ############### Run model ###############
    alphafold_model = alphafold_model.eval()
    for _ in range(num_recycle+1):
        with torch.no_grad():
            ret = alphafold_model(input_batch, non_ensembled_batch, ensemble_representations=ensemble_representations)
        non_ensembled_batch = {
            'prev_pos': ret['structure_module']['final_atom_positions'],
            'prev_msa_first_row': ret['representations']['msa_first_row'],
            'prev_pair': ret['representations']['pair']
        }
    
    return ret
