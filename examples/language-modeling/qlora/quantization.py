import torch


def qdq_weight_asym(weight, bits=4, quantile=1.0, return_int=False, **kwargs):
    """Quant and dequant tensor with asym schema.

    Args:
        weight:  input weight
        bits (int, optional): bits. Defaults to 4.
        quantile (float, optional): percentile of clip. Defaults to 1.0.
        return_int (bool, optional): Choose return fp32 or int8/uint8 data.
                                     Defaults to False.

    Returns:
        output: qdq weight
    """
    maxq = torch.tensor(2**bits - 1)
    zeros = torch.zeros(weight.shape[0], device=weight.device)
    wmin = torch.minimum(weight.min(1)[0], zeros)
    wmax = torch.maximum(weight.max(1)[0], zeros)
    wmin = wmin * quantile
    wmax = wmax * quantile
    tmp = (wmin == 0) & (wmax == 0)
    wmin[tmp] = -1
    wmax[tmp] = +1
    scale = (wmax - wmin) / maxq
    zp = torch.round(-wmin / scale)
    scale.unsqueeze_(dim=-1)
    zp.unsqueeze_(dim=-1)
    weight.div_(scale)
    weight.round_()
    weight.add_(zp)
    weight.clamp_(0, maxq)
    keep_scale = kwargs.get("double_quant", False)
    if return_int or keep_scale:
        return weight.to(dtype=torch.bfloat16), scale.to(dtype=torch.bfloat16), zp.to(dtype=torch.bfloat16)
    weight.sub_(zp)
    return weight.mul_(scale)


def qdq_weight_sym(weight, bits=4, quantile=1.0, return_int=False, full_range=False, **kwargs):
    """Quant and dequant tensor with sym schema.

    Args:
        weight : input weight
        bits (int, optional): bits. Defaults to 4.
        quantile (float, optional): percentile of clip. Defaults to 1.0.
        return_int (bool, optional): Choose return fp32 or int8/uint8 data.
                                     Defaults to False.
        full_range (bool, optional): Choose sym range whether use -2**(bits-1).
                For example: 4 bit
                    scale = amax / 8 if full_range else amax / 7
                    If True, scale = -scale if abs(min)> abs(max) else scale
                    Defaults to False.

    Returns:
        output: qdq weight
    """
    # assert bits > 1, "symmetric scheme only supports bits > 1"
    maxq = torch.tensor(2 ** (bits - 1) - 1).to(weight.device)
    minq = torch.tensor(-(2 ** (bits - 1))).to(weight.device)
    if bits == 1:  # pragma: no cover
        maxq = torch.tensor(2 ** (bits - 1))
        minq = torch.tensor(2 ** (bits - 1) - 1)
    max_val = torch.max(weight, 1)[0]
    min_val = torch.min(weight, 1)[0]
    flip_flag = torch.abs(max_val) > torch.abs(min_val)
    wmax = torch.max(torch.abs(max_val), torch.abs(min_val))
    wmax = wmax * quantile
    tmp = wmax == 0
    wmax[tmp] = torch.tensor(1, dtype=wmax.dtype, device=wmax.device)
    if full_range:
        # use -8, 8 to make sure amax is not changed after fake quant
        scale = wmax / (-minq)
        # set negative scale with flip_flag
        scale = torch.where(flip_flag, -scale, scale)
    else:
        scale = wmax / maxq
    scale.unsqueeze_(dim=-1)
    weight.div_(scale)
    weight.round_()
    weight.clamp_(minq, maxq)
    keep_scale = kwargs.get("double_quant", False)
    if return_int or keep_scale:
        return weight, scale, None
    return weight.mul_(scale)


def qdq_weight_actor(weight, bits, scheme, quantile=1.0, dtype="int", return_int=False, full_range=False, **kwargs):
    """Quant and dequant tensor per channel. It is an in-place op.

    Args:
        weight : input weight
        bits (int, optional): bits. Defaults to 4.
        quantile (float, optional): percentile of clip. Defaults to 1.0.
        dtype (str, optional): select from int, nf4, fp4. Defaults to int.
        return_int (bool, optional): Choose return fp32 or int8/uint8 data.
                                     Defaults to False.
        full_range (bool, optional): Choose sym range whether use -2**(bits-1).

    Returns:
        output: qdq weight
    """
    assert bits > 0, "bits should be larger than 0"

    if scheme == "sym":
        return qdq_weight_sym(weight, bits, quantile, return_int, full_range, **kwargs)
    else:
        return qdq_weight_asym(weight, bits, quantile, return_int, **kwargs)