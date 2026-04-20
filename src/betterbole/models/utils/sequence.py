import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence

from betterbole.models.utils.general import DNN


class LocalActivationUnit(nn.Module):
    """
    来自DIN的序列局部注意力，输出L个打分，forward输入目标物品和用户序列
    """

    def __init__(self, embedding_dim=128, hidden_units=(128, 32), activation='sigmoid', dropout_rate=0, dice_dim=3, use_bn=False):
        super(LocalActivationUnit, self).__init__()

        self.dnn = DNN(inputs_dim=4 * embedding_dim,
                       hidden_units=hidden_units,
                       activation=activation,
                       dropout_rate=dropout_rate,
                       dice_dim=dice_dim,
                       use_bn=use_bn)
        self.dense = nn.Linear(hidden_units[-1], 1)

    def forward(self, query, user_behavior):
        """
        :param query: B D
        :param user_behavior: B L D
        """
        B, L, D = user_behavior.shape
        queries = query.reshape(B, 1, D).expand(-1, L, -1)

        attention_input = torch.cat([queries, user_behavior, queries - user_behavior, queries * user_behavior],
                                    dim=-1)  # as the source code, subtraction simulates verctors' difference
        attention_output = self.dnn(attention_input)
        attention_score = self.dense(attention_output)  # [B, T, 1]
        return attention_score


class SequencePoolingLayer(nn.Module):
    """
    优雅重构版的 SequencePoolingLayer
    支持 'sum', 'mean', 'max' 池化，自动处理 Padding 掩码。
    """
    def __init__(self, mode='mean', eps=1e-8):
        super().__init__()
        if mode not in ['sum', 'mean', 'max']:
            raise ValueError("mode 必须是 ['sum', 'mean', 'max'] 之一")
        self.mode = mode
        self.eps = eps

    def forward(self, seq, seq_len):
        """
        Args:
            seq: [B, L, E]
            seq_len: [B, 1] 或 [B]
        Returns: [B, 1, D] 的 Pooling 结果
        """
        # 1. 确保 seq_len 是 [B, 1] 形状，方便后续广播
        if seq_len.dim() == 1:
            seq_len = seq_len.unsqueeze(-1)

        _, L, _ = seq.shape
        mask = (torch.arange(L, device=seq.device)[None, :] < seq_len).unsqueeze(-1)
        # 3. 执行 Pooling
        if self.mode == 'max':
            seq_masked = torch.where(mask, seq, torch.tensor(-1e9, device=seq.device, dtype=seq.dtype))
            return torch.max(seq_masked, dim=1, keepdim=True)[0]

        elif self.mode == 'sum':
            seq_masked = seq * mask.float()
            return torch.sum(seq_masked, dim=1, keepdim=True)

        elif self.mode == 'mean':
            seq_masked = seq * mask.float()
            sum_pool = torch.sum(seq_masked, dim=1, keepdim=True).squeeze(1)
            return sum_pool / seq_len.float().clamp_min(self.eps)



class AttentionSequencePoolingLayer(nn.Module):
    def __init__(self, embedding_dim=128, att_hidden_units=(128, 32), att_activation='dice'):
        super(AttentionSequencePoolingLayer, self).__init__()
        self.local_att = LocalActivationUnit(hidden_units=att_hidden_units, embedding_dim=embedding_dim,
                                             activation=att_activation,
                                             dropout_rate=0, use_bn=False)
    def forward(self, query, keys, keys_len):
        """
        :param query: B D
        :param keys: B L D
        :param keys_len: B
        :return:
        """
        B, L, D = keys.shape
        # Mask
        keys_masks = torch.arange(L, device=keys_len.device, dtype=keys_len.dtype).repeat(B, 1)  # [B, T]
        keys_masks = keys_masks < keys_len.view(-1, 1)  # 0, 1 mask
        keys_masks = keys_masks.unsqueeze(1)  # [B, 1, T]

        score = self.local_att(query, keys)  # [B, L, 1]
        score = torch.transpose(score, 1, 2)  # [B, 1, L]
        paddings = torch.zeros_like(score)
        outputs = torch.where(keys_masks, score, paddings)  # [B, 1, D]
        outputs = torch.matmul(outputs, keys)  # [B, 1, D]
        return outputs.squeeze(1)


class KMaxPooling(nn.Module):
    """K Max pooling that selects the k biggest value along the specific axis.

      Input shape
        -  nD tensor with shape: ``(batch_size, ..., input_dim)``.

      Output shape
        - nD tensor with shape: ``(batch_size, ..., output_dim)``.

      Arguments
        - **k**: positive integer, number of top elements to look for along the ``axis`` dimension.

        - **axis**: positive integer, the dimension to look for elements.

     """

    def __init__(self, k, axis, device='cpu'):
        super(KMaxPooling, self).__init__()
        self.k = k
        self.axis = axis
        self.to(device)

    def forward(self, inputs):
        if self.axis < 0 or self.axis >= len(inputs.shape):
            raise ValueError("axis must be 0~%d,now is %d" %
                             (len(inputs.shape) - 1, self.axis))

        if self.k < 1 or self.k > inputs.shape[self.axis]:
            raise ValueError("k must be in 1 ~ %d,now k is %d" %
                             (inputs.shape[self.axis], self.k))

        out = torch.topk(inputs, k=self.k, dim=self.axis, sorted=True)[0]
        return out


class AGRUCell(nn.Module):
    """ Attention based GRU (AGRU)

        Reference:
        -  Deep Interest Evolution Network for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1809.03672, 2018.
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(AGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # (W_ir|W_iz|W_ih)
        self.weight_ih = nn.Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.register_parameter('weight_ih', self.weight_ih)
        # (W_hr|W_hz|W_hh)
        self.weight_hh = nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.register_parameter('weight_hh', self.weight_hh)
        if bias:
            # (b_ir|b_iz|b_ih)
            self.bias_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_ih', self.bias_ih)
            # (b_hr|b_hz|b_hh)
            self.bias_hh = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_hh', self.bias_hh)
            for tensor in [self.bias_ih, self.bias_hh]:
                nn.init.zeros_(tensor, )
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

    def forward(self, inputs, hx, att_score):
        gi = F.linear(inputs, self.weight_ih, self.bias_ih)
        gh = F.linear(hx, self.weight_hh, self.bias_hh)
        i_r, _, i_n = gi.chunk(3, 1)
        h_r, _, h_n = gh.chunk(3, 1)

        reset_gate = torch.sigmoid(i_r + h_r)
        # update_gate = torch.sigmoid(i_z + h_z)
        new_state = torch.tanh(i_n + reset_gate * h_n)

        att_score = att_score.view(-1, 1)
        hy = (1. - att_score) * hx + att_score * new_state
        return hy


class AUGRUCell(nn.Module):
    """ Effect of GRU with attentional update gate (AUGRU)

        Reference:
        -  Deep Interest Evolution Network for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1809.03672, 2018.
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(AUGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # (W_ir|W_iz|W_ih)
        self.weight_ih = nn.Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.register_parameter('weight_ih', self.weight_ih)
        # (W_hr|W_hz|W_hh)
        self.weight_hh = nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.register_parameter('weight_hh', self.weight_hh)
        if bias:
            # (b_ir|b_iz|b_ih)
            self.bias_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_ih', self.bias_ih)
            # (b_hr|b_hz|b_hh)
            self.bias_hh = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_ih', self.bias_hh)
            for tensor in [self.bias_ih, self.bias_hh]:
                nn.init.zeros_(tensor, )
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

    def forward(self, inputs, hx, att_score):
        gi = F.linear(inputs, self.weight_ih, self.bias_ih)
        gh = F.linear(hx, self.weight_hh, self.bias_hh)
        i_r, i_z, i_n = gi.chunk(3, 1)
        h_r, h_z, h_n = gh.chunk(3, 1)

        reset_gate = torch.sigmoid(i_r + h_r)
        update_gate = torch.sigmoid(i_z + h_z)
        new_state = torch.tanh(i_n + reset_gate * h_n)

        att_score = att_score.view(-1, 1)
        update_gate = att_score * update_gate
        hy = (1. - update_gate) * hx + update_gate * new_state
        return hy


class DynamicGRU(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, gru_type='AGRU'):
        super(DynamicGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        if gru_type == 'AGRU':
            self.rnn = AGRUCell(input_size, hidden_size, bias)
        elif gru_type == 'AUGRU':
            self.rnn = AUGRUCell(input_size, hidden_size, bias)

    def forward(self, inputs, att_scores=None, hx=None):
        if not isinstance(inputs, PackedSequence) or not isinstance(att_scores, PackedSequence):
            raise NotImplementedError("DynamicGRU only supports packed input and att_scores")

        inputs, batch_sizes, sorted_indices, unsorted_indices = inputs
        att_scores, _, _, _ = att_scores

        max_batch_size = int(batch_sizes[0])
        if hx is None:
            hx = torch.zeros(max_batch_size, self.hidden_size,
                             dtype=inputs.dtype, device=inputs.device)

        outputs = torch.zeros(inputs.size(0), self.hidden_size,
                              dtype=inputs.dtype, device=inputs.device)

        begin = 0
        for batch in batch_sizes:
            new_hx = self.rnn(
                inputs[begin:begin + batch],
                hx[0:batch],
                att_scores[begin:begin + batch])
            outputs[begin:begin + batch] = new_hx
            hx = new_hx
            begin += batch
        return PackedSequence(outputs, batch_sizes, sorted_indices, unsorted_indices)