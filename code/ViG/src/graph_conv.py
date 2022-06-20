import torch
import torch.nn as nn


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer

    act = act.lower()
    if act == "relu":
        layer = nn.ReLU(inplace)
    elif act == "leakyrelu":
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == "prelu":
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == "gelu":
        layer = nn.GELU()
    else:
        raise NotImplementedError("activation layer [%s] is not found" % act)
    return layer


def norm_layer(norm, nc):
    # normalization layer 2d
    norm = norm.lower()
    if norm == "batch":
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm == "instance":
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm)
    return layer


class BasicConv(nn.Sequential):
    def __init__(self, channels, act="relu", norm=None, bias=True, drop=0.0):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Conv2d(channels[i - 1], channels[i], 1, bias=bias))
            if act is not None and act.lower() != "none":
                m.append(act_layer(act))
            if norm is not None and norm.lower() != "none":
                m.append(norm_layer(norm, channels[-1]))
            if drop > 0:
                m.append(nn.Dropout2d(drop))

        super(BasicConv, self).__init__(*m)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def batched_index_select(x, idx):
    """
    Args:
        X: B, C, N, 1
        e_index: B, N, K

    Returns:
        B, C, N, k
    """

    batch_size, num_dims, num_vertices = x.shape[:3]

    k = idx.shape[-1]
    idx_base = (
        torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_vertices
    )

    idx = idx + idx_base
    idx = idx.contiguous().view(-1)

    x = x.transpose(2, 1)

    feature = x.contiguous().view(batch_size * num_vertices, -1)[idx, :]
    feature = (
        feature.view(batch_size, num_vertices, k, num_dims)
        .permute(0, 3, 1, 2)
        .contiguous()
    )

    return feature  # B, C, N, K


class MRConv2d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """

    def __init__(self, in_channels, out_channels, act="relu", norm=None, bias=True):
        super(MRConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index):
        shape = x.shape
        x = x.flatten(2).unsqueeze(-1)

        x_i = batched_index_select(x, edge_index[1])
        x_j = batched_index_select(x, edge_index[0])

        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        b, c, n, _ = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _)

        return self.nn(x).reshape(shape)
