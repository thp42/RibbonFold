from . import quat_affine, r3, all_atom, utils
from .common_modules import *


def squared_difference(x, y):
    return torch.square(x - y)

class InvariantPointAttention(nn.Module):
    """Invariant Point attention module.
    
    The high-level idea is that this attention module works over a set of points
    and associated orientations in 3D space (e.g. protein residues).
    
    Each residue outputs a set of queries and keys as points in their local
    reference frame.  The attention is then defined as the euclidean distance
    between the queries and keys in the global frame.
    
    Jumper et al. (2021) Suppl. Alg. 22 "InvariantPointAttention"
    """
    
    def __init__(self, single_dim=384, pair_dim=128, 
        num_head=12, num_scalar_qk=16, num_point_qk=4,
        num_scalar_v=16, num_point_v=8, 
        zero_init=True, dist_epsilon=1e-8):
        """Initialize.
        """
        super().__init__()
        
        assert num_scalar_qk > 0
        assert num_point_qk > 0
        assert num_point_v > 0
        
        dist_epsilon = small_value(dist_epsilon)
        
        self.single_dim = single_dim
        self.pair_dim = pair_dim
        self.num_head = num_head
        self.num_scalar_qk = num_scalar_qk
        self.num_point_qk = num_point_qk
        self.num_scalar_v = num_scalar_v
        self.num_point_v = num_point_v
        self.zero_init = zero_init
        self.dist_epsilon = dist_epsilon
        
        self.q_scalar = Linear(single_dim, num_head * num_scalar_qk)
        self.kv_scalar = Linear(single_dim, num_head * (num_scalar_v + num_scalar_qk))
        self.q_point_local = Linear(single_dim, num_head * 3 * num_point_qk)
        self.kv_point_local = Linear(single_dim, num_head * 3 * (num_point_qk + num_point_v))
        
        # softplus^{-1} (1)
        self.trainable_point_weights = nn.Parameter(torch.ones([num_head]) * np.log(np.exp(1.) - 1.))
        
        self.attention_2d = Linear(pair_dim, num_head)
        
        concat_dim = num_head * num_scalar_v + num_head * num_point_v * 3 + num_head * num_point_v + num_head * pair_dim
        self.output_projection = Linear(concat_dim, single_dim, initializer='zeros' if zero_init else 'linear')
    
    def forward(self, inputs_1d, inputs_2d, mask, affine, local_mask_2d=None):
        """Compute geometry-aware attention.
        Args:
          inputs_1d: [N_res, C] single representation
          inputs_2d: [N_res, N_res, c_m] pair representation
          mask: [N_res, 1] mask to indicate which elements of inputs_1d participate in the attention.
          affine: QuatAffine object describing the position and orientation of every element in inputs_1d.
        
        Returns:
          Transformation of the input embedding.
        """
        assert inputs_1d.ndim == 2
        assert inputs_2d.ndim == 3
        assert mask.ndim == 2
        assert inputs_1d.shape[1] == self.single_dim
        assert inputs_2d.shape[2] == self.pair_dim
        
        num_residues, _ = inputs_1d.shape
        
        # Improve readability by removing a large number of 'self's.
        num_head = self.num_head # 12
        num_scalar_qk = self.num_scalar_qk # 16
        num_point_qk = self.num_point_qk # 4
        num_scalar_v = self.num_scalar_v # 16
        num_point_v = self.num_point_v # 8
        num_output = self.single_dim # 384
        
        # Construct scalar queries of shape:
        # [num_query_residues, num_head, num_points]
        q_scalar = self.q_scalar(inputs_1d) # [N_res, H * scalar_qk]
        q_scalar = torch.reshape(q_scalar, [num_residues, num_head, num_scalar_qk]) # [N_res, H, scalar_qk]
        
        # Construct scalar keys/values of shape:
        # [num_target_residues, num_head, num_points]
        kv_scalar = self.kv_scalar(inputs_1d) # [N_res, H * (scalar_v + scalar_qk)]
        kv_scalar = torch.reshape(kv_scalar,[num_residues, num_head, num_scalar_v + num_scalar_qk]) # [N_res, H, scalar_v + scalar_qk]
        # k_scalar: [N_res, H, scalar_qk]
        # v_scalar: [N_res, H, scalar_v]
        k_scalar, v_scalar = torch.split(kv_scalar, split_size_or_sections=[num_scalar_qk, num_scalar_v], dim=-1) 
        
        # Construct query points of shape:
        # [num_residues, num_head, num_point_qk]
        
        # First construct query points in local frame.
        q_point_local = self.q_point_local(inputs_1d) # [N_res, H * 3 * point_qk]
        q_point_local = torch.split(q_point_local, q_point_local.shape[-1] // 3, dim=-1) # [ [N_res, H * point_qk], [N_res, H * point_qk], [N_res, H * point_qk] ]
        # Project query points into global frame.
        q_point_global = affine.apply_to_point(q_point_local, extra_dims=1)
        # Reshape query point for later use.
        q_point = [torch.reshape(x, [num_residues, num_head, num_point_qk]) for x in q_point_global] # [N_res, H, point_qk] X ??
        
        # Construct key and value points.
        # Key points have shape [num_residues, num_head, num_point_qk]
        # Value points have shape [num_residues, num_head, num_point_v]
        
        # Construct key and value points in local frame.
        kv_point_local = self.kv_point_local(inputs_1d) # [N_res, H * 3 * (point_qk + point_v)]
        kv_point_local = torch.split(kv_point_local, kv_point_local.shape[-1] // 3, dim=-1) # [N_res, H * (point_qk + point_v)] X 3
        # Project key and value points into global frame.
        kv_point_global = affine.apply_to_point(kv_point_local, extra_dims=1)
        # [N_res, H, (point_qk + point_v)] X ??
        kv_point_global = [torch.reshape(x, [num_residues, num_head, (num_point_qk + num_point_v)]) for x in kv_point_global]
        # Split key and value points.
        # k_point: [N_res, H, point_qk] X ??
        # v_point: [N_res, H, point_v] X ??
        k_point, v_point = list( zip(*[ torch.split(x, [num_point_qk, num_point_v], dim=-1) for x in kv_point_global ]) )
        
        # We assume that all queries and keys come iid from N(0, 1) distribution
        # and compute the variances of the attention logits.
        # Each scalar pair (q, k) contributes Var q*k = 1
        scalar_variance = max(num_scalar_qk, 1) * 1.
        # Each point pair (q, k) contributes Var [0.5 ||q||^2 - <q, k>] = 9 / 2
        point_variance = max(num_point_qk, 1) * 9. / 2
        
        # Allocate equal variance to scalar, point and attention 2d parts so that
        # the sum is 1.
        
        num_logit_terms = 3
        
        scalar_weights = np.sqrt(1.0 / (num_logit_terms * scalar_variance)) # 0.144
        point_weights = np.sqrt(1.0 / (num_logit_terms * point_variance)) # 0.136
        attention_2d_weights = np.sqrt(1.0 / (num_logit_terms)) # 0.577
        
        # Trainable per-head weights for points.
        trainable_point_weights = F.softplus(self.trainable_point_weights)
        point_weights *= torch.unsqueeze(trainable_point_weights, dim=1) # [H, 1]
        
        v_point = [torch.swapaxes(x, -2, -3) for x in v_point] # [H, N_res, point_v]  X ??
        q_point = [torch.swapaxes(x, -2, -3) for x in q_point] # [H, N_res, point_qk] X ??
        k_point = [torch.swapaxes(x, -2, -3) for x in k_point] # [H, N_res, point_qk] X ??
        dist2 = [
            squared_difference(qx[:, :, None, :], kx[:, None, :, :]) # [H, N_res, 1, point_qk] vs [H, 1, N_res, point_qk]
            for qx, kx in zip(q_point, k_point)
        ] # [H, N_res, N_res, point_qk] X ??
        dist2 = sum(dist2) # [H, N_res, N_res, point_qk]
        # [H, 1, 1, 1] * [H, N_res, N_res, point_qk] -> [H, N_res, N_res, point_qk] -> [H, N_res, N_res]
        attn_qk_point = -0.5 * torch.sum(point_weights[:, None, None, :] * dist2, dim=-1) # [H, N_res, N_res]
        
        v = torch.swapaxes(v_scalar, -2, -3) # [H, N_res, scalar_v]
        q = torch.swapaxes(scalar_weights * q_scalar, -2, -3) # [H, N_res, scalar_qk]
        k = torch.swapaxes(k_scalar, -2, -3) # [H, N_res, scalar_qk]
        attn_qk_scalar = torch.matmul(q, torch.swapaxes(k, -2, -1)) # [H, N_res, scalar_qk] @ [H, scalar_qk, N_res] -> [H, N_res, N_res]
        attn_logits = attn_qk_scalar + attn_qk_point # [H, N_res, N_res]


        if local_mask_2d is not None:
            inputs_2d = inputs_2d * local_mask_2d.unsqueeze(-1)

        attention_2d = self.attention_2d(inputs_2d) # [N_res, N_res, H]
        # if local_mask_2d is not None:
        #     attention_2d = attention_2d * local_mask_2d.unsqueeze(-1)

        attention_2d = torch.permute(attention_2d, [2, 0, 1]) # [H, N_res, N_res]
        attention_2d = attention_2d_weights * attention_2d   #  [H, N_res, N_res]
        attn_logits = attn_logits + attention_2d # [H, N_res, N_res]
        
        mask_2d = mask * torch.swapaxes(mask, -1, -2) # [N_res, 1] * [1, N_res] -> [N_res, N_res]
        attn_logits -= large_value(1e5) * (1. - mask_2d) # [H, N_res, N_res]
        
        # [num_head, num_query_residues, num_target_residues]
        attn = F.softmax(attn_logits, dim=-1) # [H, N_res, N_res]
        
        # [num_head, num_query_residues, num_head * num_scalar_v]
        result_scalar = torch.matmul(attn, v) # [H, N_res, N_res] @ [H, N_res, scalar_v] -> [H, N_res, scalar_v]
        
        # For point result, implement matmul manually so that it will be a float32
        # on TPU.  This is equivalent to
        # result_point_global = [jnp.einsum('bhqk,bhkc->bhqc', attn, vx)
        #                        for vx in v_point]
        # but on the TPU, doing the multiply and reduce_sum ensures the
        # computation happens in float32 instead of bfloat16.
        # [H, N_res, N_res, 1] * [H, 1, N_res, point_v] -> [H, N_res, N_res, point_v] -> [H, N_res, point_v] X ??
        result_point_global = [torch.sum(attn[:, :, :, None] * vx[:, None, :, :], axis=-2) for vx in v_point]
        
        # [num_query_residues, num_head, num_head * num_(scalar|point)_v]
        result_scalar = torch.swapaxes(result_scalar, -2, -3) # [N_res, H, scalar_v]
        result_point_global = [ torch.swapaxes(x, -2, -3) for x in result_point_global] # [N_res, H, point_v] X ??
        
        # Features used in the linear output projection. Should have the size
        # [num_query_residues, ?]
        output_features = []
        
        result_scalar = torch.reshape(result_scalar, [num_residues, num_head * num_scalar_v]) # [N_res, H * scalar_v=192] 
        output_features.append(result_scalar)
        
        result_point_global = [
            torch.reshape(r, [num_residues, num_head * num_point_v]) # [N_res, H * point_v] X ??
            for r in result_point_global]
        result_point_local = affine.invert_point(result_point_global, extra_dims=1)
        output_features.extend(result_point_local) # + [ N_res, H * point_v * 3 = 288 ? ]
        
        output_features.append(torch.sqrt(self.dist_epsilon + 
                                        torch.square(result_point_local[0]) + 
                                        torch.square(result_point_local[1]) +
                                        torch.square(result_point_local[2]))) # + [N_res, H * point_v = 96]
        
        # Dimensions: h = heads, i and j = residues,
        # c = inputs_2d channels
        # Contraction happens over the second residue dimension, similarly to how
        # the usual attention is performed.
        result_attention_over_2d = torch.einsum('hij, ijc->ihc', attn, inputs_2d) # [H, N_res, N_res], [N_res, N_res, c_m] -> [N_res, H, c_m]
        num_out = num_head * result_attention_over_2d.shape[-1] # c_m * H
        output_features.append(torch.reshape(result_attention_over_2d, [num_residues, num_out])) # + [N_res, H * c_m]
        
        final_act = torch.cat(output_features, axis=-1) # [N_res, ????]
        
        return self.output_projection(final_act)
    
def generate_new_affine(sequence_mask):
    """
    sequence_mask: [N_res, 1]
    """
    num_residues, _ = sequence_mask.shape
    quaternion = torch.tile(torch.reshape(torch.tensor([1., 0., 0., 0.]), [1, 4]),[num_residues, 1]) # [N_res, 4]
    translation = torch.zeros([num_residues, 3]) # [N_res, 3]
    return quat_affine.QuatAffine(quaternion, translation, unstack_inputs=True)

def l2_normalize(x, axis=-1, epsilon=1e-12):
    epsilon = small_value(epsilon)
    epsilon = torch.tensor(epsilon).to(x)
    return x / torch.sqrt(torch.maximum(torch.sum(x**2, dim=axis, keepdim=True), epsilon))

class MultiRigidSidechain(nn.Module):
    """Class to make side chain atoms."""
    def __init__(self, single_dim=384, num_channels=128, zero_init=True):
        super().__init__()
        self.single_dim = single_dim
        self.num_channels = num_channels
        #self.num_residual_block = num_residual_block
        self.zero_init = zero_init
        
        self.input_projection = Linear(single_dim, num_channels)
        self.input_projection_1 = Linear(single_dim, num_channels)
        
        # For convinience load the AF2 parameters
        self.resblock1 = Linear(num_channels, num_channels, initializer='relu')
        self.resblock2 = Linear(num_channels, num_channels, initializer='zeros' if zero_init else 'linear')
        self.resblock1_1 = Linear(num_channels, num_channels, initializer='relu')
        self.resblock2_1 = Linear(num_channels, num_channels, initializer='zeros' if zero_init else 'linear')

        self.unnormalized_angles = Linear(num_channels, 14)

    def forward(self, affine, act0, initial_act0, aatype):
        """Predict side chains using multi-rigid representations.

        Args:
          affine: The affines for each residue (translations in angstroms).
          act0: [N_res, single_dim]
          initial_act0: [N_res, single_dim]
          aatype: [N_res]
        
        Returns:
          Dict containing atom positions and frames (in angstroms).
        """
        assert act0.ndim == 2
        assert initial_act0.ndim == 2
        assert aatype.ndim == 1
        assert act0.shape[1] == initial_act0.shape[1] == self.single_dim
        assert aatype.shape[0] == act0.shape[0] == initial_act0.shape[0]
        num_res = aatype.shape[0]

        act = 0
        act = act + self.input_projection( torch.relu(act0) )
        act = act + self.input_projection_1( torch.relu(initial_act0) )

        # Residual blocks
        old_act = act
        act = self.resblock1( torch.relu(act) )
        act = self.resblock2( torch.relu(act) )
        act = act + old_act
        old_act = act
        act = self.resblock1_1( torch.relu(act) )
        act = self.resblock2_1( torch.relu(act) )
        act = act + old_act

        # Map activations to torsion angles. Shape: (num_res, 14).
        unnormalized_angles = self.unnormalized_angles(torch.relu(act)) # [N_res, 14]
        unnormalized_angles = torch.reshape(unnormalized_angles, [num_res, 7, 2]) # [N_res, 7, 2]
        angles = l2_normalize(unnormalized_angles, axis=-1) # [N_res, 7, 2]

        outputs = {
            'angles_sin_cos': angles,  # [N, 7, 2]
            'unnormalized_angles_sin_cos': unnormalized_angles,  # [N, 7, 2]
        }

        # Map torsion angles to frames.
        backb_to_global = r3.rigids_from_quataffine(affine)

        # Jumper et al. (2021) Suppl. Alg. 24 "computeAllAtomCoordinates"

        #print("aatype:", aatype.device)
        #print("angles:", angles.device)
        #print("backb_to_global:", backb_to_global.rot.xx.device, backb_to_global.trans.x.device)
        # r3.Rigids with shape (N, 8).
        all_frames_to_global = all_atom.torsion_angles_to_frames(
            aatype,
            backb_to_global,
            angles)

        # Use frames and literature positions to create the final atom coordinates.
        # r3.Vecs with shape (N, 14).
        pred_positions = all_atom.frames_and_literature_positions_to_atom14_pos(aatype, all_frames_to_global)

        outputs.update({
            'atom_pos': pred_positions,  # r3.Vecs (N, 14)
            'frames': all_frames_to_global,  # r3.Rigids (N, 8)
        })
        return outputs

class FoldIteration(nn.Module):
    """A single iteration of the main structure module loop.

    Jumper et al. (2021) Suppl. Alg. 20 "StructureModule" lines 6-21

    First, each residue attends to all residues using InvariantPointAttention.
    Then, we apply transition layers to update the hidden representations.
    Finally, we use the hidden representations to produce an update to the
    affine of each residue.
    """
    def __init__(self, single_dim=384, pair_dim=128, 
                    num_head=12, num_scalar_qk=16, 
                    num_point_qk=4, num_scalar_v=16, 
                    num_point_v=8, dropout=0.1,
                     sc_num_channels=128,
                    zero_init=True, deterministic=False):
        super().__init__()
        self.single_dim = single_dim
        self.pair_dim = pair_dim
        self.dropout = dropout
        self.sc_num_channels = sc_num_channels
        self.zero_init = zero_init
        self.deterministic = deterministic
    
        self.invariant_point_attention = InvariantPointAttention(
            single_dim=single_dim, 
            pair_dim=pair_dim, 
            num_head=num_head, 
            num_scalar_qk=num_scalar_qk, 
            num_point_qk=num_point_qk,
            num_scalar_v=num_scalar_v, 
            num_point_v=num_point_v, 
            zero_init=zero_init, 
            dist_epsilon=small_value(1e-8))
        
        self.attention_layer_norm = LayerNorm(single_dim)
        self.transition_layer_norm = LayerNorm(single_dim)

        self.transition = Linear(single_dim, single_dim, initializer='relu')
        self.transition_1 = Linear(single_dim, single_dim, initializer='relu')
        self.transition_2 = Linear(single_dim, single_dim, initializer='zeros' if zero_init else 'linear')

        self.affine_update = Linear(single_dim, 6, initializer='zeros' if zero_init else 'linear')

        self.rigid_sidechain = MultiRigidSidechain(single_dim=single_dim, num_channels=sc_num_channels, zero_init=zero_init)

    def forward(self, activations, sequence_mask,  update_affine, initial_act, static_feat_2d, aatype, local_mask_2d=None):
        """
        activations: dict
            -- affine: QuatAffine object
            -- act: [N_res, single_dim]
        sequence_mask: [N_res, 1]
        update_affine: bool
        initial_act: [N_res, single_dim]
        static_feat_2d: [N_res, N_res, pair_dim]
        aatype: [N_res]
        """
        assert 'affine' in activations
        assert 'act' in activations
        assert activations['act'].ndim == 2
        assert sequence_mask.ndim == 2
        assert initial_act.ndim == 2
        assert static_feat_2d.ndim == 3
        assert aatype.ndim == 1
        assert initial_act.shape[1] == self.single_dim
        assert static_feat_2d.shape[2] == self.pair_dim
        
        act = activations['act']
        affine = quat_affine.QuatAffine.from_tensor(activations['affine']).to(act.device)
        
        attn = self.invariant_point_attention(inputs_1d=act, inputs_2d=static_feat_2d, mask=sequence_mask, affine=affine, local_mask_2d=local_mask_2d)
        
        act = act + attn

        if not self.deterministic:
            act = F.dropout(act, p=self.dropout, training=self.training)

        act = self.attention_layer_norm(act)
        
        # Transition
        input_act = act
        act = self.transition(act)
        act = torch.relu(act)
        act = self.transition_1(act)
        act = torch.relu(act)
        act = self.transition_2(act)
        act = act + input_act

        if not self.deterministic:
            act = F.dropout(act, p=self.dropout, training=self.training)

        act = self.transition_layer_norm(act)

        if update_affine:
            # This block corresponds to
            # Jumper et al. (2021) Alg. 23 "Backbone update"
            affine_update_size = 6

            # Affine update
            affine_update = self.affine_update(act)

            affine = affine.pre_compose(affine_update)

        sc = self.rigid_sidechain(affine.scale_translation(10.0), act, initial_act, aatype)

        outputs = {'affine': affine.to_tensor(), 'sc': sc}

        # Stop gradients
        affine = affine.apply_rotation_tensor_fn(lambda x: x.detach())

        new_activations = {
            'act': act,
            'affine': affine.to_tensor()
        }
        return new_activations, outputs

class StructureModule(nn.Module):
    """StructureModule as a network head.

    Jumper et al. (2021) Suppl. Alg. 20 "StructureModule"
    """

    def __init__(self, single_dim=384, pair_dim=128, num_layer=8, sc_num_channels=128,
                compute_loss=True, zero_init=True, deterministic=False):
        super().__init__()

        self.single_dim = single_dim
        self.pair_dim = pair_dim
        self.num_layer = num_layer
        self.sc_num_channels = sc_num_channels
        self.compute_loss = compute_loss
        self.zero_init = zero_init
        self.deterministic = deterministic
        
        self.single_layer_norm = LayerNorm(single_dim)
        self.initial_projection = Linear(single_dim, single_dim)
        self.fold_iteration = FoldIteration(
                    single_dim=single_dim, 
                    pair_dim=pair_dim, 
                    num_head=12, 
                    num_scalar_qk=16, 
                    num_point_qk=4, 
                    num_scalar_v=16, 
                    num_point_v=8, 
                    sc_num_channels=sc_num_channels,
                    dropout=0.1,
                    zero_init=zero_init, 
                    deterministic=deterministic)
        self.pair_layer_norm = LayerNorm(pair_dim)

    def forward(self, representations, batch, local_ipa=False, recycle_id=None):
        """
        representations: dict
            -- single: [N_res, single_dim]
            -- pair: [N_res, N_res, pair_dim]
        batch: dict
            -- seq_mask: [N_res]
            -- aatype: [N_res]
            -- atom14_atom_exists: [N_res, 14]
            -- atom37_atom_exists: [N_res, 37]
            -- residx_atom37_to_atom14: [N_res, 37]
        local_ipa: bool
            whether mask pair representation by local intervals
        """
        assert representations['single'].ndim == 2
        assert representations['pair'].ndim == 3
        assert batch['seq_mask'].ndim == 1
        assert batch['aatype'].ndim == 1
        assert batch['atom14_atom_exists'].ndim == 2
        assert batch['atom37_atom_exists'].ndim == 2
        assert batch['residx_atom37_to_atom14'].ndim == 2
        assert representations['single'].shape[1] == self.single_dim
        assert representations['pair'].shape[2] == self.pair_dim
        
        ret = {}
        
        output = self.generate_affines(representations, batch, local_ipa=local_ipa, recycle_id=recycle_id)
        
        representations['structure_module'] = output['act']
        representations['structure_module1'] = output['act1']

        position_scale = 10.0
        ret['traj'] = output['affine'] * torch.tensor([1.] * 4 + [position_scale] * 3).to(output['affine'].device)

        ret['sidechains'] = output['sc']

        atom14_pred_positions = r3.vecs_to_tensor(output['sc']['atom_pos'])[-1]
        # atom14_pred_positions = r3.vecs_to_tensor(output['sc']['atom_pos'])[0]
        ret['final_atom14_positions'] = atom14_pred_positions  # (N, 14, 3)
        ret['final_atom14_mask'] = batch['atom14_atom_exists']  # (N, 14)

        atom37_pred_positions = all_atom.atom14_to_atom37(atom14_pred_positions, batch)
        atom37_pred_positions *= batch['atom37_atom_exists'][:, :, None]
        ret['final_atom_positions'] = atom37_pred_positions  # [N, 37, 3]

        ret['final_atom_mask'] = batch['atom37_atom_exists']  # [N, 37]
        ret['final_affines'] = ret['traj'][-1]
        # ret['final_affines'] = ret['traj'][0]

        if self.compute_loss:
            return ret
        else:
            no_loss_features = ['final_atom_positions', 'final_atom_mask']
            no_loss_ret = {k: ret[k] for k in no_loss_features}
            return no_loss_ret

    def generate_affines(self, representations, batch, local_ipa=False, recycle_id=None):
        """Generate predicted affines for a single chain.

        Jumper et al. (2021) Suppl. Alg. 20 "StructureModule"

        This is the main part of the structure module - it iteratively applies
        folding to produce a set of predicted residue positions.

        Args:
            representations: Representations dictionary.
            batch: Batch dictionary.

        Returns:
            A dictionary containing residue affines and sidechain positions.
        """
        sequence_mask = batch['seq_mask'][:, None] # [N_res, 1]

        act = self.single_layer_norm(representations['single']) # [N_res, single_dim]
        
        initial_act = act
        act = self.initial_projection(act) # [N_res, single_dim]
        
        affine = generate_new_affine(sequence_mask).to(act.device)
        
        activations = {
            'act': act,
            'affine': affine.to_tensor()
        }

        act_2d = self.pair_layer_norm(representations['pair']) # [N_res, N_res, pair_dim]

        def create_pair_mask(N_res, k=3):
            N_single_res = N_res // 5   # pentamer
            block_size = N_single_res // k
            mask = torch.zeros((N_res, N_res), dtype=torch.float32)

            # for i in range(k):
            #     start = i * block_size
            #     end = start + block_size if i < k - 1 else N_res
            #     mask[start:end, start:end] = 1
            for i in range(N_res):
                for j in range(N_res):
                    # if residue i and j are in the same block of single seq, mask is 1
                    reminder_i = i % N_single_res
                    reminder_j = j % N_single_res
                    if reminder_i // block_size == reminder_j // block_size:
                        # if abs(reminder_i - reminder_j) <= block_size:
                        mask[i, j] = 1

            return mask


        def run_fold_iter(module, act, affine, sequence_mask,  update_affine, initial_act, static_feat_2d, aatype):
            """
            The activations dict must be splited into two variables: act & affine
            """
            activations = { 'act': act, 'affine': affine }
            activations, output = module(activations, sequence_mask,  update_affine, initial_act, static_feat_2d, aatype)
            return activations['act'], activations['affine'], output['affine'], output['sc']
        
        outputs = []
        act1 = activations['act'].clone()

        for layer_idx in range(self.num_layer):
            # if self.training:
            #     a1, a2, a3, a4 = checkpoint(run_fold_iter, 
            #                                self.fold_iteration, 
            #                                activations['act'], 
            #                                activations['affine'], 
            #                                sequence_mask,  
            #                                True, 
            #                                initial_act, 
            #                                act_2d, 
            #                                batch['aatype'])
            #     activations = { 'act': a1, 'affine': a2 }
            #     outputs.append( { 'affine': a3.clone(), 'sc': a4 } )
            # else:

            # if layer_idx < 4:
            if local_ipa:
                # local_k_schedule = np.linspace(8, 1, 24)
                # print('layer: ', layer_idx, local_ipa)
                # local_k_schedule = [8,8,8,7,7,7,6,6,6,5,5,5,4,4,4,3,3,3,2,2,2,1,1,1]
                local_k_schedule = [4,4,4,4,3,3,3,3,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1]
                # local_k_schedule = [5,5,5,4,4,4,3,3,3,2,2,2,1,1,1,1, 1, 1, 1, 1, 1, 1, 1, 1]
                total_module = recycle_id * 8 + layer_idx
                local_k = local_k_schedule[total_module]
                N_res = representations['pair'].shape[0]
                mask_2d = create_pair_mask(N_res, k=local_k).to(act_2d.device)
                # masked_act_2d = act_2d * mask_2d.unsqueeze(-1)
                mask_rate = mask_2d.sum() / (mask_2d.shape[0] * mask_2d.shape[1])
                # print(f'recycle {recycle_id} module {layer_idx}: local k is {local_k}, mask rate is {mask_rate}')
            else:
                # masked_act_2d = act_2d
                mask_2d = None


            activations, output = self.fold_iteration(
               activations = activations,
               initial_act = initial_act,
               static_feat_2d = act_2d,         #masked_act_2d,
               sequence_mask = sequence_mask,
               update_affine = True,
               aatype = batch['aatype'],
               local_mask_2d = mask_2d
            )
            outputs.append(output)

        output = utils.tree_multimap(lambda *x: torch.stack(x), *outputs)
        # Include the activations in the output dict for use by the LDDT-Head.
        output['act'] = activations['act']

        output['act1'] = act1

        return output
    



