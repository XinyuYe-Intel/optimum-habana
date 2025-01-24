import copy
from typing import Any, Dict, Optional, Union
import logging

import torch

from torch.nn.parameter import Parameter

from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING, PeftType
from peft.tuners.lora import LoraLayer, LoraModel
from peft.utils.other import transpose

from .quantization import qdq_weight_actor
from .functions import matmul_4bit


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
        self.dqw = (w_4bit-zp)*scales
        
        self.pack(w_4bit, scales, zp)
        self.quant_state = {"scales": self.scales, "zero_point": self.qzeros}
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

    def quantize(self, device="hpu"):
        return self._quantize(device)


class Linear4bit(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        compute_dtype=None,
        quant_type="int4",
        quant_storage=torch.int32,
        device=None,
        dtype=None,
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
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.compute_dtype = compute_dtype
        self.compute_type_is_set = False
        self.quant_type = quant_type
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
            for k, v in self.weight.quant_state.items():
                destination[prefix + "weight." + k] = v if keep_vars else v.detach()

    def forward(self, x: torch.Tensor):
        if not isinstance(self.weight, Params4bit):
            self.weight = Params4bit(
                self.weight.data,
                requires_grad=False,
                quant_type=self.quant_type,
                quant_storage=self.quant_storage,
                module=self,
            )
            self.weight.quantize()

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
        return matmul_4bit(x, self.weight.data, bias=bias, quant_state=self.weight.quant_state).to(inp_dtype)

    # def to(self, *args, **kwargs):
    #     device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
    #     def convert(t):
    #         try:
    #             if convert_to_format is not None and t.dim() in (4, 5):
    #                 return t.to(
    #                     device,
    #                     dtype if t.is_floating_point() or t.is_complex() else None,
    #                     non_blocking,
    #                     memory_format=convert_to_format,
    #                 )
    #             return t.to(
    #                 device,
    #                 dtype if t.is_floating_point() or t.is_complex() else None,
    #                 non_blocking,
    #             )
    #         except NotImplementedError as e:
    #             if str(e) == "Cannot copy out of meta tensor; no data!":
    #                 raise NotImplementedError(
    #                     f"{e} Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to() "
    #                     f"when moving module from meta to a different device."
    #                 ) from None
    #             else:
    #                 raise
    #     self.weight.data = convert(self.weight)
    #     self.bias.data = None if self.bias is None else convert(self.bias)


class LoraLinear4bit(torch.nn.Module, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    # def merge(self, safe_merge: bool = False) -> None:
    #     """Merge the active adapter weights into the base weights.

    #     Args:
    #         safe_merge (`bool`, *optional*):
    #             If True, the merge operation will be performed in a copy of the original weights and check for NaNs
    #             before merging the weights. This is useful if you want to check if the merge operation will produce
    #             NaNs. Defaults to `False`.
    #     """
    #     if self.merged:
    #         print(f"Already following adapters were merged {','.join(self.merged_adapters)}. "
    #               f"You are now additionally merging {','.join(self.active_adapters)}.")
    #     w_dequant = torch.zeros(
    #         self.out_features,
    #         self.in_features,
    #         dtype=list(self.lora_A.values())[0].weight.dtype,
    #     )
    #     qbits.dequantize_packed_weight(
    #         self.weight.data,
    #         w_dequant,
    #         True,
    #         self.compute_dtype,
    #         self.weight_dtype,
    #         self.scale_dtype,
    #     )
    #     w_data = w_dequant
    #     for active_adapter in self.active_adapters:
    #         if active_adapter in self.lora_A.keys():
    #             if safe_merge:
    #                 # Note that safe_merge will be slower than the normal merge
    #                 # because of the copy operation.
    #                 orig_weights = w_data.clone()
    #                 orig_weights += self.get_delta_weight(active_adapter)

    #                 if not torch.isfinite(orig_weights).all():
    #                     raise ValueError(
    #                         f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken")

    #                 w_data = orig_weights
    #             else:
    #                 w_data += self.get_delta_weight(active_adapter)

    #     weight = qbits.quantize_to_packed_weight(
    #         w_data,
    #         True,
    #         self.blocksize,
    #         self.compute_dtype,
    #         self.weight_dtype,
    #         self.scale_dtype,
    #         False if self.scheme == "sym" else True,
    #     )

    #     self.weight = ParamsQBits(
    #         data=weight,
    #         requires_grad=False,
    #         quant_state={"scheme": self.scheme},
    #         blocksize=self.blocksize,
    #         compress_statistics=self.compress_statistics,
    #         quant_dtype=self.weight_dtype,
    #         scale_dtype=self.scale_dtype,
    #     )

    # def unmerge(self) -> None:
    #     if not self.merged:
    #         print("Already unmerged. Nothing to do.")
    #         return

    #     w_dequant = torch.zeros(
    #         self.out_features,
    #         self.in_features,
    #         dtype=list(self.lora_A.values())[0].weight.dtype,
    #     )
    #     qbits.dequantize_packed_weight(
    #         self.weight.data,
    #         w_dequant,
    #         True,
    #         self.compute_dtype,
    #         self.weight_dtype,
    #         self.scale_dtype,
    #     )

    #     w_data = w_dequant
    #     while len(self.merged_adapters) > 0:
    #         active_adapter = self.merged_adapters.pop()
    #         if active_adapter in self.lora_A.keys():
    #             w_data -= self.get_delta_weight(active_adapter)

    #     weight = qbits.quantize_to_packed_weight(
    #         w_data,
    #         True,
    #         self.blocksize,
    #         self.compute_dtype,
    #         self.weight_dtype,
    #         self.scale_dtype,
    #         False if self.scheme == "sym" else True,
    #     )

    #     self.weight = ParamsQBits(
    #         data=weight,
    #         requires_grad=False,
    #         quant_state={"scheme": self.scheme},
    #         blocksize=self.blocksize,
    #         compress_statistics=self.compress_statistics,
    #         quant_dtype=self.weight_dtype,
    #         scale_dtype=self.scale_dtype,
    #     )

    def get_delta_weight(self, adapter) -> torch.Tensor:
        return (transpose(
            self.lora_B[adapter].weight @ self.lora_A[adapter].weight,
            False,
        ) * self.scaling[adapter])

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                if not self.use_dora[active_adapter]:
                    result = result + lora_B(lora_A(dropout(x))) * scaling
                else:
                    x = dropout(x)
                    result = result + self.lora_magnitude_vector[active_adapter](
                        x,
                        lora_A=lora_A,
                        lora_B=lora_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                    )

            result = result.to(torch_result_dtype)

        return result


class QLoraModel(LoraModel):
    _create_new_module_ = LoraModel._create_new_module

    def _create_new_module(self, lora_config, adapter_name, target, **kwargs):
        if isinstance(target, Linear4bit):
            if kwargs["fan_in_fan_out"]:
                print("fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                      "Setting fan_in_fan_out to False.")
                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
            kwargs["compute_dtype"] = target.compute_dtype
            kwargs["quant_storage"] = target.quant_storage
            kwargs["quant_state"] = target.quant_state
            new_module = LoraLinear4bit(target, adapter_name, **kwargs)
        else:
            new_module = QLoraModel._create_new_module_(lora_config, adapter_name, target, **kwargs)
        return new_module


PEFT_TYPE_TO_MODEL_MAPPING[PeftType.LORA] = QLoraModel


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