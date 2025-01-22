import copy
from typing import Any, Dict, Optional
import logging

import torch

from quantization import qdq_weight_actor
from functions import matmul_4bit


class Params4bit(torch.nn.Parameter):
    def __new__(
        cls,
        data: Optional[torch.Tensor] = None,
        requires_grad=False,  # quantized weights should be frozen by default
        quant_state: Dict[str, Any] = None,
        blocksize: int = 64,
        quant_type: str = "int4",
        quant_storage: torch.dtype = torch.int32,
        module: Optional["Linear4bit"] = None,
        quantized: bool = False,
    ) -> "Params4bit":
        if data is None:
            data = torch.empty(0)

        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        self.blocksize = blocksize
        self.quant_type = quant_type
        self.quant_state = quant_state
        self.quant_storage = quant_storage
        self.quantized = quantized
        self.data = data
        self.module = module
        self.bits = 4
        self.wf = torch.tensor(list(range(0, 32, self.bits)), dtype=torch.int32).unsqueeze(0)
        return self

    def __getstate__(self):
        state = self.__dict__.copy()
        state["data"] = self.data
        state["requires_grad"] = self.requires_grad
        return state

    def __setstate__(self, state):
        self.requires_grad = state["requires_grad"]
        self.blocksize = state["blocksize"]
        self.quant_type = state["quant_type"]
        self.quant_state = state["quant_state"]
        self.data = state["data"]
        self.quant_storage = state["quant_storage"]
        self.quantized = state["quantized"]
        self.module = state["module"]

    def __deepcopy__(self, memo):
        new_instance = type(self).__new__(type(self))
        state = self.__getstate__()
        new_instance.__setstate__(state)
        new_instance.quant_state = copy.deepcopy(state["quant_state"])
        new_instance.data = copy.deepcopy(state["data"])
        return new_instance

    def __copy__(self):
        new_instance = type(self).__new__(type(self))
        state = self.__getstate__()
        new_instance.__setstate__(state)
        return new_instance

    def _quantize(self, device):
        w = self.data.contiguous().to(device)
        w_4bit, scales, zp = qdq_weight_actor(w, 4, scheme="asym", return_int=True)
        zp = torch.zeros_like(scales) if zp is None else zp
        self.quant_state = {"scales": scales.clone(), "zero_point": zp.clone()}
        self.dqw = (w_4bit-zp)*scales
        
        self.pack(w_4bit, scales, zp)
        if self.module is not None:
            self.module.quant_state = self.quant_state
        self.quantized = True
        return self

    def pack(self, int_weight, scales, zp):
        """Pack weight and zero point."""
        logging.debug("Packing for HPU")
        scales = scales.T.contiguous()
        qzeros = zp.T.contiguous()
        data = int_weight.T.contiguous()

        self.scales = scales

        # weights and zp are on device from unpack, need to load to cpu for packing
        new_data = self.pack_tensor(data.cpu())
        self.data = new_data.to("hpu")

        new_qzeros = self.pack_tensor(qzeros.cpu())
        self.qzeros = new_qzeros.to("hpu")

    def pack_tensor(self, input):
        """Pack tensor."""
        normal = input.to(self.quant_storage)
        bits = self.bits
        q = torch.zeros((normal.shape[0], normal.shape[1] * bits // 32), dtype=self.quant_storage)
        i = 0
        col = 0
        while col < q.shape[1]:
            for j in range(i, i + (32 // bits)):
                q[:, col] |= normal[:, j] << (bits * (j - i))
            i += 32 // bits
            col += 1
        q = q.to(self.quant_storage)
        return q

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)

        if device is not None and device.type == "hpu" and not self.quantized:
            return self._quantize(device)
        else:
            if self.quant_state is not None:
                self.quant_state.to(device)

            new_param = Params4bit(
                super().to(device=device, dtype=dtype, non_blocking=non_blocking),
                requires_grad=self.requires_grad,
                quant_state=self.quant_state,
                blocksize=self.blocksize,
                quant_type=self.quant_type,
                quant_storage=self.quant_storage,
            )

            return new_param


class Linear4bit(torch.nn.Linear):
    """
    This class is the base module for the 4-bit quantization algorithm presented in [QLoRA](https://arxiv.org/abs/2305.14314).
    QLoRA 4-bit linear layers uses blockwise k-bit quantization under the hood, with the possibility of selecting various
    compute datatypes such as FP4 and NF4.

    In order to quantize a linear layer one should first load the original fp16 / bf16 weights into
    the Linear4bit module, then call `quantized_module.to("cuda")` to quantize the fp16 / bf16 weights.

    Example:

    ```python
    import torch
    import torch.nn as nn

    import bitsandbytes as bnb
    from bnb.nn import Linear4bit

    fp16_model = nn.Sequential(
        nn.Linear(64, 64),
        nn.Linear(64, 64)
    )

    quantized_model = nn.Sequential(
        Linear4bit(64, 64),
        Linear4bit(64, 64)
    )

    quantized_model.load_state_dict(fp16_model.state_dict())
    quantized_model = quantized_model.to(0) # Quantization happens here
    ```
    """

    def __init__(
        self,
        input_features,
        output_features,
        bias=True,
        compute_dtype=None,
        quant_type="int4",
        quant_storage=torch.int32,
        device=None,
    ):
        """
        Initialize Linear4bit class.

        Args:
            input_features (`str`):
                Number of input features of the linear layer.
            output_features (`str`):
                Number of output features of the linear layer.
            bias (`bool`, defaults to `True`):
                Whether the linear class uses the bias term as well.
        """
        super().__init__(input_features, output_features, bias, device)
        self.weight = Params4bit(
            self.weight.data,
            requires_grad=False,
            quant_type=quant_type,
            quant_storage=quant_storage,
            module=self,
        )
        # self.persistent_buffers = []  # TODO consider as way to save quant state
        self.compute_dtype = compute_dtype
        self.compute_type_is_set = False
        self.quant_state = None
        self.quant_storage = quant_storage

    def set_compute_type(self, x):
        if x.dtype in [torch.float32, torch.bfloat16]:
            # the input is in a dtype that is safe to compute in, we switch
            # to this type for speed and stability
            self.compute_dtype = x.dtype
        elif x.dtype == torch.float16:
            # we take the compoute dtype passed into the layer
            if self.compute_dtype == torch.float32 and (x.numel() == x.shape[-1]):
                # single batch inference with input torch.float16 and compute_dtype float32 -> slow inference when it could be fast
                # warn the user about this
                logging.warning(
                    "Input type into Linear4bit is torch.float16, but bnb_4bit_compute_dtype=torch.float32 (default). This will lead to slow inference.",
                )
            if self.compute_dtype == torch.float32 and (x.numel() != x.shape[-1]):
                logging.warning(
                    "Input type into Linear4bit is torch.float16, but bnb_4bit_compute_dtype=torch.float32 (default). This will lead to slow inference or training speed.",
                )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        """
        save weight and bias,
        then fill state_dict with components of quant_state
        """
        super()._save_to_state_dict(destination, prefix, keep_vars)  # saving weight and bias

        if getattr(self.weight, "quant_state", None) is not None:
            for k, v in self.weight.quant_state.as_dict(packed=True).items():
                destination[prefix + "weight." + k] = v if keep_vars else v.detach()

    def forward(self, x: torch.Tensor):
        if not self.weight.quantized:
            self.weight.to(x.device)

        # weights are cast automatically as Int8Params, but the bias has to be cast manually
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        if not self.compute_type_is_set:
            self.set_compute_type(x)
            self.compute_type_is_set = True

        inp_dtype = x.dtype
        if self.compute_dtype is not None:
            x = x.to(self.compute_dtype)

        bias = None if self.bias is None else self.bias.to(self.compute_dtype)
        # import pdb; pdb.set_trace()
        return matmul_4bit(x, self.weight, bias=bias, quant_state=self.weight.quant_state).to(inp_dtype)

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        def convert(t):
            try:
                if convert_to_format is not None and t.dim() in (4, 5):
                    return t.to(
                        device,
                        dtype if t.is_floating_point() or t.is_complex() else None,
                        non_blocking,
                        memory_format=convert_to_format,
                    )
                return t.to(
                    device,
                    dtype if t.is_floating_point() or t.is_complex() else None,
                    non_blocking,
                )
            except NotImplementedError as e:
                if str(e) == "Cannot copy out of meta tensor; no data!":
                    raise NotImplementedError(
                        f"{e} Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to() "
                        f"when moving module from meta to a different device."
                    ) from None
                else:
                    raise
        self.weight.data = convert(self.weight)
        self.bias.data = None if self.bias is None else convert(self.bias)


if __name__ == "__main__":
    def print_msg(msg, info=""):
        print("="*20 + info + "="*20)
        print(msg)
        print("="*(40+len(info)) + "\n")
    import habana_frameworks.torch
    torch.manual_seed(42)
    weight = torch.randn(128, 32, dtype=torch.bfloat16).to("hpu")
    weightq = Params4bit(weight.data.clone())
    weightq.to("hpu")
    print_msg(weight, info="weight reference")
    weightq_deq = weightq.dqw
    print_msg(weightq_deq, info="qdq weight")
    print_msg(torch.abs(weightq.dqw-weight).max(), info="error of qdq weight and weight reference")
    weightq_hpu = torch.ops.hpu.convert_from_uint4(weightq.data, weightq.scales, weightq.qzeros, weight.dtype).t()
    print_msg(weightq_hpu, info="hpu qdq weight")
    print_msg(torch.abs(weightq.dqw-weightq_hpu).max(), info="error of hpu qdq weight and qdq weight")