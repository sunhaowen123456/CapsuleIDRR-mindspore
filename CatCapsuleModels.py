import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import ops
from mindspore.common.initializer import initializer, XavierUniform


class CapsuleNetwork(nn.Cell):
    def __init__(self, conf=None):
        super(CapsuleNetwork, self).__init__()
        if conf.need_elmo:
            hidden_size = 705
        else:
            hidden_size = 705
        config = {}
        config['keep_prob'] = conf.keep_prob  # 0
        config['hidden_size'] = hidden_size  # 705
        config['output_atoms'] = conf.output_atoms  # 16
        config['r'] = 40
        config['num_routing'] = conf.num_routing  # 3

        self.conf = conf
        self.hidden_size = config['hidden_size']
        self.drop = nn.Dropout(p=config['keep_prob'])

        self.r = config['r']
        self.s_cnum = conf.four_or_eleven  # numclass
        self.output_atoms = self.conf.output_atoms  # 16

        self.input_dim = self.r  # 40
        self.input_atoms = self.hidden_size  # 705
        self.output_dim = self.s_cnum  # numclass
        self.capsule_weights_1 = ms.Parameter(ms.Tensor(np.zeros((self.r, self.hidden_size, self.s_cnum * self.output_atoms), dtype=np.float32)))
        self.capsule_weights_2 = ms.Parameter(ms.Tensor(np.zeros((self.r, self.hidden_size, self.s_cnum * self.output_atoms), dtype=np.float32)))
        # weights[40,705,numclass*16]
        self.h = conf.num_routing_head
        self.d_k = self.output_atoms // self.h
        self.init_weights()

    def construct(self, arg1):  # [128,40,705]
        votes_reshaped_1 = self.capsule(arg1, self.capsule_weights_1)  # u(j|i)[128,40,numclass,16]
        input_shape = self.sentence_embedding.shape
        logit_shape = np.stack([input_shape[0], self.input_dim, self.output_dim])  # [128,40,numclass]

        self.activation, _, _ = self.routing_single(votes_reshaped_1, logit_shape, num_dims=4)  # v:[128,numclass,16]
        self.logits = self.get_logits()  # [128,numclass]

    def capsule(self, input, capsule_weights):  # [128,40,705] [40,705,numclass*16]
        self.sentence_embedding = input
        dropout_emb = self.drop(self.sentence_embedding)
        input_tiled = ops.unsqueeze(dropout_emb, -1).tile((1, 1, 1, self.s_cnum * self.output_atoms))  # [128,40,705,numclass*16]
        votes = ops.sum(input_tiled * capsule_weights, dim=2)  # [128,40,numclass*16]
        votes_reshaped = ops.reshape(votes, [-1, self.input_dim, self.s_cnum, self.output_atoms])

        return votes_reshaped  # u(j|i)[128,40,numclass,16]

    def get_logits(self):
        logits = ops.norm(self.activation, dim=-1)
        return logits

    def _routing(self, votes_trans, logits, r_t_shape, num_dims, input_dim):
        route = ops.softmax(logits, axis=1)   # c[128,40,numclass]
        preactivate_unrolled = route * votes_trans  # [16,128,40,numclass]
        preact_trans = preactivate_unrolled.permute(r_t_shape)  # [128,40,numclass,16]
        preactivate = ops.sum(preact_trans, dim=1)  # [128,numclass,16]
        activation = self._squash(preactivate)  # v [128,numclass,16]
        act_3d = activation.unsqueeze(1)  # [128,1,numclass,16]
        tile_shape = np.ones(num_dims, dtype=np.int64)
        tile_shape[1] = input_dim
        tile_shape = tuple(tile_shape)
        act_replicated = act_3d.tile(tile_shape)  # [128,40,numclass,16]

        return route, activation, act_replicated

    def routing_single(self, votes_arg, logit_shape, num_dims):  # u(j|i)[128,40,numclass,16] [128,40,numclass]
        votes_t_shape = [3, 0, 1, 2]
        for i in range(num_dims - 4):
            votes_t_shape += [i + 4]
        r_t_shape = [1, 2, 3, 0]
        for i in range(num_dims - 4):
            r_t_shape += [i + 4]

        votes_trans = votes_arg.permute(votes_t_shape)  # [16,128,40,numclass]
        logits = ms.Parameter(ms.Tensor(np.zeros((logit_shape[0], logit_shape[1], logit_shape[2]), dtype=np.float32)))  # b[128,40,numclass]
        activations = []
        route = None
        for iteration in range(self.conf.num_routing):  # 3
            route, activation, act_replicated = self._routing(votes_trans, logits, r_t_shape, num_dims,
                                                              self.input_dim)  # c,v,v_repeat
            distances = ops.sum(votes_arg * act_replicated, dim=3)  # alpha:u(j|i)*v_repeat[128,40,numclass]
            logits = logits + distances  # 更新b
            activations.append(activation)  # v [128,numclass,16]

        return activations[self.conf.num_routing - 1], logits, route  # route:c[128,40,numclass]

    def _squash(self, input_tensor):
        norm = ops.norm(input_tensor, dim=2, keepdim=True)
        norm_squared = norm * norm
        return (input_tensor / norm) * (norm_squared / (0.5 + norm_squared))

    def init_weights(self):
        self.capsule_weights_1.set_data(initializer(XavierUniform(), [self.r, self.hidden_size, self.s_cnum * self.output_atoms], ms.float32))
        self.capsule_weights_2.set_data(initializer(XavierUniform(), [self.r, self.hidden_size, self.s_cnum * self.output_atoms], ms.float32))
        self.capsule_weights_1.requires_grad = True
        self.capsule_weights_2.requires_grad = True

    def _margin_loss(self, labels, raw_logits, margin=0.4, downweight=0.5):
        logits = raw_logits - 0.5
        positive_cost = labels * (logits < margin).float() * ((logits - margin) ** 2)
        negative_cost = (1 - labels) * (logits > -margin).float() * ((logits + margin) ** 2)
        return 0.5 * positive_cost + downweight * 0.5 * negative_cost

    def loss(self, label):
        loss_val = self._margin_loss(label, self.logits)
        loss_val = ops.mean(loss_val)

        return 1000 * loss_val
