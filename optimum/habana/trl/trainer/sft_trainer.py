# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import transformers
from accelerate import PartialState
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BaseImageProcessor,
    DataCollator,
    DataCollatorWithFlattening,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainingArguments,
    is_wandb_available,
)
from transformers.data.data_collator import DataCollatorForLanguageModeling as TransformersDataCollatorForLanguageModeling
from transformers.data.data_collator import DataCollatorMixin, pad_without_fast_tokenizer_warning
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_peft_available
from trl import SFTTrainer
from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    maybe_convert_to_chatml,
    pack_dataset,
    truncate_dataset,
)
from trl.trainer.utils import (
    ConstantLengthDataset,
    generate_model_card,
    get_comet_experiment_url,
    pad,
    peft_module_casting_to_bf16,
)

from ... import GaudiConfig, GaudiTrainer
from ...utils import warn0
from .sft_config import GaudiSFTConfig


if is_peft_available():
    import peft
    from peft import PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

if is_wandb_available():
    import wandb


@dataclass
class DataCollatorForLanguageModeling(DataCollatorMixin):
    """
    Data collator used for language modeling data. Inputs are dynamically padded to the maximum length of a batch if
    they are not all of the same length.

    Args:
        pad_token_id (`int`):
            Token ID to use for padding.
        completion_only_loss (`bool`, *optional*, defaults to `True`):
            When the input contains a completion mask (`completion_mask`), the labels are set to -100 for the tokens
            that are not in the completion.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            Type of Tensor to return. Only `"pt"` is currently supported.
    """

    pad_token_id: int
    completion_only_loss: bool = True
    return_tensors: str = "pt"

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        # Convert to tensor
        input_ids = [torch.tensor(example["input_ids"]) for example in examples]
        attention_mask = [torch.ones_like(ids) for ids in input_ids]
        labels = [torch.tensor(example["input_ids"]) for example in examples]
        if self.completion_only_loss and "completion_mask" in examples[0]:
            completion_mask = [torch.tensor(example["completion_mask"]) for example in examples]

        # Pad
        output = {}
        output["input_ids"] = pad(input_ids, padding_value=self.pad_token_id, padding_side="right")
        output["attention_mask"] = pad(attention_mask, padding_value=0, padding_side="right")
        output["labels"] = pad(labels, padding_value=-100, padding_side="right")
        if self.completion_only_loss and "completion_mask" in examples[0]:
            completion_mask = pad(completion_mask, padding_value=0, padding_side="right")
            output["labels"][completion_mask == 0] = -100  # mask everything that is not in the completion

        return output


class BucketedDataCollatorForLanguageModeling(TransformersDataCollatorForLanguageModeling):
    """Gaudi-specific data collator that uses bucketing for more efficient padding."""

    def _get_bucketed_len(self, examples):
        max_sentence_len = max([len(k["input_ids"]) for k in examples])
        if max_sentence_len > self.buckets[-1]:
            self.buckets = np.append(self.buckets, max_sentence_len)
            curr_bucket = max_sentence_len
        else:
            curr_bucket = self.buckets[np.argmin(np.where(max_sentence_len <= self.buckets))]
        return curr_bucket

    # copied from https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/data/data_collator.py#L758
    # change is pad_to_multiple_of=self.pad_to_multiple_of -> pad_to_multiple_of=bucketed_len
    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        from collections.abc import Mapping

        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            bucketed_len = self._get_bucketed_len(examples)
            batch = pad_without_fast_tokenizer_warning(
                self.tokenizer,
                examples,
                return_tensors="pt",
                pad_to_multiple_of=bucketed_len,
            )
        else:
            assert False, "This path has not been implemented/tested yet"

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch


class GaudiSFTTrainer(SFTTrainer, GaudiTrainer):
    """
    Gaudi-compatible Supervised Fine-Tuning (SFT) Trainer.

    Adapted from https://github.com/huggingface/trl/blob/cd6b3de/trl/trainer/sft_trainer.py
    Key differences from upstream SFTTrainer:
    - Uses `GaudiTrainer` as base instead of `Trainer`
    - Accepts `gaudi_config` parameter for Gaudi hardware optimizations
    - Supports bucketing via `num_buckets` for Gaudi hardware efficiency
    - Supports `pad_max` for static shapes on Gaudi (configured via `GaudiSFTConfig`)
    - Uses `GaudiSFTConfig` instead of `SFTConfig`
    """

    _tag_names = ["trl", "sft"]

    def __init__(
        self,
        model: Union[str, nn.Module, PreTrainedModel] = None,
        args: Optional[GaudiSFTConfig] = None,
        gaudi_config: GaudiConfig = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
            None,
            None,
        ),
        optimizer_cls_and_kwargs: Optional[tuple[type[torch.optim.Optimizer], dict[str, Any]]] = None,
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional["PeftConfig"] = None,
        formatting_func: Optional[Union[Callable[[dict], str], Callable[[dict], list[str]]]] = None,
        num_buckets: Optional[int] = -1,
    ):
        """
        Initialize GaudiSFTTrainer.

        Args:
            model: Model to train. Can be a string (model path) or a PreTrainedModel.
            args: GaudiSFTConfig with training configuration.
            gaudi_config: Gaudi-specific configuration for hardware optimizations.
            data_collator: Custom data collator.
            train_dataset: Training dataset.
            eval_dataset: Evaluation dataset.
            processing_class: Tokenizer or processor. If None, loaded from model config.
            compute_loss_func: Custom loss function.
            compute_metrics: Custom metrics function.
            callbacks: List of trainer callbacks.
            optimizers: Optimizer and scheduler tuple.
            optimizer_cls_and_kwargs: Optimizer class and kwargs (requires transformers>=4.47.0).
            preprocess_logits_for_metrics: Preprocess logits before metrics computation.
            peft_config: PEFT configuration for parameter-efficient fine-tuning.
            formatting_func: Function to format dataset examples.
            num_buckets: Number of buckets for bucketed data collation (Gaudi-specific).
                > 0 enables bucketing, <= 0 disables it.
        """
        if num_buckets > 0:
            assert data_collator is None, (
                "For bucketing (num_buckets > 0), we only support data_collator=None "
                "(later it becomes BucketedDataCollatorForLanguageModeling)"
            )

        # Args
        model_id = model if isinstance(model, str) else getattr(getattr(model, "config", None), "_name_or_path", "")
        if args is None:
            model_name = model_id.split("/")[-1] if model_id else "model"
            warn0(f"No `GaudiSFTConfig` passed, using `output_dir={model_name}-SFT`.")
            args = GaudiSFTConfig(f"{model_name}-SFT")
        elif isinstance(args, TrainingArguments) and not isinstance(args, GaudiSFTConfig):
            dict_args = args.to_dict()
            dict_args["hub_token"] = args.hub_token  # to_dict hides the hub_token
            dict_args.pop("push_to_hub_token", None)
            args = GaudiSFTConfig(**dict_args)

        # Handle the tokenizer / processing_class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model_id)
            if getattr(processing_class, "pad_token", None) is None:
                processing_class.pad_token = processing_class.eos_token

        if args.eos_token is not None:
            eos_token = args.eos_token
            eos_token_id = processing_class.convert_tokens_to_ids(eos_token)
            if eos_token_id is None:
                raise ValueError(
                    f"The specified `eos_token` ('{eos_token}') is not found in the vocabulary of the given "
                    f"`processing_class` ({processing_class.__class__.__name__}). Ensure that the `eos_token` exists "
                    "in the vocabulary before using it as an EOS token."
                )
            processing_class.eos_token_id = eos_token_id

        # Model
        if args.model_init_kwargs is not None and not isinstance(model, str):
            warn0(
                "You passed model_init_kwargs to the `GaudiSFTConfig`, but your model is already instantiated. "
                "The `model_init_kwargs` will be ignored."
            )
        if isinstance(model, str):
            model = self._create_model_from_path(model, args)

        # PEFT configuration and model wrapping
        if peft_config is not None:
            model = self._prepare_peft_model(model, peft_config, args)

        # Data collator
        if args.padding_free:
            if data_collator is not None:
                raise ValueError("Passing a custom data collator is not supported when using padding-free.")
            if args.packing:
                warnings.warn(
                    "You are passing `packing=True` and `padding_free=True` which is not recommended. Please refer "
                    "to the documentation to understand why this is not recommended."
                )
            if model.config._attn_implementation != "flash_attention_2":
                warnings.warn(
                    "Padding-free training is enabled, but the attention implementation is not set to "
                    "'flash_attention_2'. Padding-free training flattens batches into a single sequence, and "
                    "'flash_attention_2' is the only known attention mechanism that reliably supports this."
                )
            if args.per_device_train_batch_size == 1:
                warnings.warn(
                    "You are using a per_device_train_batch_size of 1 with padding-free training. Using a batch size "
                    "of 1 annihilate the benefits of padding-free training. Please consider increasing the batch size "
                    "to at least 2."
                )
            data_collator = DataCollatorWithFlattening()

        if num_buckets <= 0:
            # Only set completion_only_loss and default data collator when not using bucketing
            if args.completion_only_loss is None:
                first_example = next(iter(train_dataset)) if train_dataset is not None else {}
                self.completion_only_loss = "prompt" in first_example
            else:
                self.completion_only_loss = args.completion_only_loss

            if data_collator is None and not args.padding_free:
                # Get the pad token: if not provided, use the one from the processing class or the eos token
                pad_token = (
                    args.pad_token
                    or getattr(processing_class, "pad_token", None)
                    or getattr(processing_class, "eos_token", None)
                )
                pad_token_id = processing_class.convert_tokens_to_ids(pad_token) if pad_token else None
                if pad_token_id is None:
                    raise ValueError(
                        f"The specified `pad_token` ('{pad_token}') is not found in the vocabulary of the given "
                        f"`processing_class` ({processing_class.__class__.__name__}). Ensure that the `pad_token` "
                        "exists in the vocabulary before using it as a padding token."
                    )
                data_collator = DataCollatorForLanguageModeling(pad_token_id, self.completion_only_loss)

        # Dataset
        preprocess_dataset = args.dataset_kwargs is None or not args.dataset_kwargs.get("skip_prepare_dataset", False)
        if preprocess_dataset:
            if train_dataset is not None:
                train_dataset = self._prepare_dataset(
                    train_dataset, processing_class, args, args.packing, formatting_func, "train"
                )
            if eval_dataset is not None:
                packing = args.packing if args.eval_packing is None else args.eval_packing
                if isinstance(eval_dataset, dict):
                    eval_dataset = {
                        key: self._prepare_dataset(dataset, processing_class, args, packing, formatting_func, key)
                        for key, dataset in eval_dataset.items()
                    }
                else:
                    eval_dataset = self._prepare_dataset(
                        eval_dataset, processing_class, args, packing, formatting_func, "eval"
                    )

        # Initialize metrics tracking
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0

        # Initialize the GaudiTrainer
        super_init_kwargs = {}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super_init_kwargs["optimizer_cls_and_kwargs"] = optimizer_cls_and_kwargs
        else:
            if optimizer_cls_and_kwargs is not None:
                warnings.warn(
                    "The `optimizer_cls_and_kwargs` argument is only available for `transformers>=4.47.0`. "
                    "The default optimizer will be used. "
                    "Remove the `optimizer_cls_and_kwargs` or upgrade to `transformers>=4.47.0`."
                )

        GaudiTrainer.__init__(
            self,
            model=model,
            args=args,
            gaudi_config=gaudi_config,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            **super_init_kwargs,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if self.args.max_steps > 0 and args.packing:
            warn0(
                "You passed `packing=True` to the GaudiSFTTrainer/GaudiSFTConfig, and you are training your model "
                "with `max_steps` strategy. The dataset will be iterated until the `max_steps` are reached.",
                state=self.accelerator.state,
            )
            self.train_dataset.infinite = True
        elif self.args.max_steps == -1 and args.packing:
            self.train_dataset.infinite = False

        # Bucketing (Gaudi-specific)
        if num_buckets > 0:
            train_dataloader = self.get_train_dataloader()
            batched_sentence_lengths = [batch["input_ids"].shape[1] for batch in train_dataloader]
            buckets = self._get_buckets(batched_sentence_lengths, num_buckets=num_buckets)
            self.data_collator = BucketedDataCollatorForLanguageModeling(tokenizer=processing_class, mlm=False)
            self.data_collator.buckets = buckets

    def _create_model_from_path(self, model_path: str, args: GaudiSFTConfig) -> PreTrainedModel:
        """Creates a model from a path or model identifier."""
        model_init_kwargs = args.model_init_kwargs or {}
        # Handle torch dtype
        torch_dtype = model_init_kwargs.get("torch_dtype")
        if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
            pass  # torch_dtype is already a torch.dtype or "auto" or None
        elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
            torch_dtype = getattr(torch, torch_dtype)
            model_init_kwargs["torch_dtype"] = torch_dtype
        else:
            raise ValueError(
                "Invalid `torch_dtype` passed to `GaudiSFTConfig`. Expected either 'auto' or a string representing "
                f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
            )
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_init_kwargs)
        return model

    def _prepare_peft_model(self, model: PreTrainedModel, peft_config: Any, args: GaudiSFTConfig) -> PreTrainedModel:
        """Prepares a model for PEFT training."""
        if not is_peft_available():
            raise ImportError("To use PeftModel, you need to install the `peft` library.")

        if not isinstance(peft_config, PeftConfig):
            raise ValueError(
                f"Expected PeftConfig object but got {type(peft_config)}. If you want to use the PeftModel, you need "
                "to pass a PeftConfig object to the GaudiSFTTrainer."
            )

        if isinstance(model, PeftModel):
            return model

        # Handle quantized models (QLoRA)
        is_qlora = getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False)

        is_sharded_qlora = False
        if getattr(model, "is_loaded_in_4bit", False):
            # Check if model is sharded (FSDP/DS-Zero3)
            for _, param in model.named_parameters():
                if param.__class__.__name__ == "Params4bit":
                    is_sharded_qlora = param.data.device.type in {"cpu", "meta"}
                    break

        # Prepare model for kbit training if needed
        if is_qlora and not is_sharded_qlora:
            model = self._prepare_model_for_kbit_training(model, args)
            # Disable gradient checkpointing as it's handled by prepare_model_for_kbit_training
            args = dataclasses.replace(args, gradient_checkpointing=False)
        elif args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Create PEFT model
        if (
            version.parse(peft.__version__) >= version.parse("0.12")  # autocast_adapter_dtype introduced in 0.12
            and getattr(model, "is_loaded_in_4bit", False)
            and is_sharded_qlora
        ):
            model = get_peft_model(model, peft_config, autocast_adapter_dtype=False)
        else:
            model = get_peft_model(model, peft_config)

        # Handle bf16 casting for 4-bit models
        if args.bf16 and getattr(model, "is_loaded_in_4bit", False) and not is_sharded_qlora:
            peft_module_casting_to_bf16(model)

        return model

    def _prepare_model_for_kbit_training(self, model: PreTrainedModel, args: GaudiSFTConfig) -> PreTrainedModel:
        """Prepares a quantized model for kbit training."""
        prepare_model_kwargs = {
            "use_gradient_checkpointing": args.gradient_checkpointing,
            "gradient_checkpointing_kwargs": args.gradient_checkpointing_kwargs or {},
        }
        return prepare_model_for_kbit_training(model, **prepare_model_kwargs)

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GaudiSFTConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        return model

    def _prepare_dataset(
        self,
        dataset: Union[Dataset, IterableDataset],
        processing_class: Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin],
        args: GaudiSFTConfig,
        packing: bool,
        formatting_func: Optional[Callable[[dict], str]],
        dataset_name: str,
    ) -> Union[Dataset, IterableDataset]:
        """
        Prepares dataset for training/evaluation.

        Adapted from upstream SFTTrainer._prepare_dataset with Gaudi-specific additions:
        - `pad_max` support: when `args.pad_max` is True, pads sequences to `max_length` during
          tokenization for static shapes on Gaudi hardware.
        """
        # Convert the dataset to an IterableDataset if it is a ConstantLengthDataset
        if isinstance(dataset, ConstantLengthDataset):
            return dataset

        # If the dataset is already preprocessed (tokenized), skip the processing steps.
        column_names = list(next(iter(dataset)).keys())
        is_processed = "input_ids" in column_names

        # Build the kwargs for the `map` function
        map_kwargs = {}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc
            map_kwargs["num_proc"] = args.dataset_num_proc

        add_special_tokens = True  # default, overridden below based on dataset type

        with PartialState().main_process_first():
            # Apply the formatting function if any
            if formatting_func is not None and is_processed:
                warnings.warn(
                    "You passed a dataset that is already processed (contains an `input_ids` field) together with a "
                    "formatting function. Therefore `formatting_func` will be ignored. Either remove the "
                    "`formatting_func` or pass a dataset that is not already processed.",
                    UserWarning,
                )

            if formatting_func is not None and not is_processed:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Applying formatting function to {dataset_name} dataset"

                def _func(example):
                    return {"text": formatting_func(example)}

                try:
                    dataset = dataset.map(_func, batched=False, **map_kwargs)
                except Exception as e:
                    warnings.warn(
                        f"Failed to apply the formatting function due to the following error: {e}. This may be "
                        "because the function is designed for batched input. Please update it to process one example "
                        "at a time (i.e., accept and return a single example). For now, we will attempt to apply the "
                        "function in batched mode, but note that batched formatting is deprecated and will be removed "
                        "in a future version.",
                        DeprecationWarning,
                    )
                    dataset = dataset.map(_func, batched=True, **map_kwargs)

            if not is_processed:
                # Convert the dataset to ChatML if needed
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Converting {dataset_name} dataset to ChatML"
                column_names = next(iter(dataset)).keys()
                dataset = dataset.map(
                    maybe_convert_to_chatml,
                    remove_columns="conversations" if "conversations" in column_names else None,
                    **map_kwargs,
                )

                # Apply the chat template if needed
                first_example = next(iter(dataset))
                if is_conversational(first_example):
                    if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                        map_kwargs["desc"] = f"Applying chat template to {dataset_name} dataset"
                    column_names = first_example.keys()
                    dataset = dataset.map(
                        apply_chat_template,
                        fn_kwargs={"tokenizer": processing_class},
                        remove_columns="messages" if "messages" in column_names else None,
                        **map_kwargs,
                    )
                    # Subsequent tokenization won't add special tokens (mostly for bos).
                    # See https://huggingface.co/blog/qgallouedec/gotchas-in-tokenizer-behavior
                    add_special_tokens = False
                else:
                    # When dataset is not conversational, add EOS token at the end of each example
                    if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                        map_kwargs["desc"] = f"Adding EOS to {dataset_name} dataset"

                    def add_eos(example, eos_token):
                        if "text" in example and not example["text"].endswith(eos_token):
                            example["text"] = example["text"] + eos_token
                        elif "completion" in example and not example["completion"].endswith(eos_token):
                            example["completion"] = example["completion"] + eos_token
                        return example

                    dataset = dataset.map(
                        add_eos,
                        fn_kwargs={"eos_token": processing_class.eos_token},
                        remove_columns="messages" if "messages" in column_names else None,
                        **map_kwargs,
                    )
                    # Subsequent tokenization will add special tokens (mostly for bos).
                    add_special_tokens = True

                # Tokenize the dataset
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

                # Gaudi-specific: support pad_max for static shape training
                pad_max = getattr(args, "pad_max", False)
                max_length = args.max_length

                def tokenize(example, processing_class, dataset_text_field, add_special_tokens, pad_max, max_length):
                    if "prompt" in example:  # prompt-completion case
                        processed_prompt = processing_class(
                            text=example["prompt"],
                            add_special_tokens=add_special_tokens,
                        )
                        processed = processing_class(
                            text=example["prompt"] + example["completion"],
                            add_special_tokens=add_special_tokens,
                            truncation=True if max_length else False,
                            padding="max_length" if pad_max and max_length else False,
                            max_length=max_length,
                        )

                        # Check if the tokenized prompt starts with the tokenized prompt+completion
                        prompt_ids = processed_prompt["input_ids"]
                        prompt_completion_ids = processed["input_ids"]
                        if not prompt_completion_ids[: len(prompt_ids)] == prompt_ids:
                            warnings.warn(
                                "Mismatch between tokenized prompt and the start of tokenized prompt+completion. "
                                "This may be due to unexpected tokenizer behavior, whitespace issues, or special "
                                "token handling."
                            )

                        # Create a completion mask
                        completion_mask = [0] * len(prompt_ids) + [1] * (
                            len(prompt_completion_ids) - len(prompt_ids)
                        )
                        processed = {**processed, "completion_mask": completion_mask}

                    else:  # language modeling case
                        processed = processing_class(
                            text=example[dataset_text_field],
                            add_special_tokens=add_special_tokens,
                            truncation=True if max_length else False,
                            padding="max_length" if pad_max and max_length else False,
                            max_length=max_length,
                        )
                    return processed

                dataset = dataset.map(
                    tokenize,
                    fn_kwargs={
                        "processing_class": processing_class,
                        "dataset_text_field": args.dataset_text_field,
                        "add_special_tokens": add_special_tokens,
                        "pad_max": pad_max,
                        "max_length": max_length,
                    },
                    **map_kwargs,
                )

            # Pack or truncate
            if packing:
                if args.max_length is None:
                    raise ValueError("When packing is enabled, `max_length` can't be `None`.")
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Packing {dataset_name} dataset"
                dataset = dataset.select_columns("input_ids")
                dataset = pack_dataset(dataset, args.max_length, map_kwargs)
            elif args.max_length is not None and not getattr(args, "pad_max", False):
                # Only truncate if pad_max is not enabled (pad_max already handles length during tokenization)
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Truncating {dataset_name} dataset"
                dataset = truncate_dataset(dataset, args.max_length, map_kwargs)

            # For Liger kernel, ensure only input_ids is present
            if getattr(args, "use_liger_kernel", False):
                dataset = dataset.select_columns("input_ids")

        return dataset

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # When using `completion_only_loss` we add a "completion_mask" column to the dataset,
        # so we need to override the default to include "completion_mask" as well.
        if self._signature_columns is None:
            self._signature_columns = ["input_ids", "attention_mask", "completion_mask"]

    def _get_buckets(self, sentence_lengths, num_buckets):
        """Compute bucket boundaries from sentence length distribution (Gaudi-specific)."""
        return np.unique(
            np.percentile(
                sentence_lengths,
                np.linspace(0, 100, num_buckets + 1),
                interpolation="lower",
            )[1:]
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies.
        """
        mode = "eval" if self.control.should_evaluate else "train"
        (loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        if mode == "train":
            if "attention_mask" in inputs:
                num_tokens_in_batch = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
            elif "position_ids" in inputs:
                local_num_tokens = torch.tensor(inputs["position_ids"].size(1), device=inputs["position_ids"].device)
                num_tokens_in_batch = self.accelerator.gather_for_metrics(local_num_tokens).sum().item()
            else:
                raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
            self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        # Compute token accuracy if we have labels and if the model is not using Liger (no logits)
        if "labels" in inputs and not getattr(self.args, "use_liger_kernel", False):
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = inputs["labels"][..., 1:].contiguous()

            # Get predictions
            predictions = shift_logits.argmax(dim=-1)

            # Create mask for non-padding tokens (assuming ignore_index is -100)
            mask = shift_labels != -100

            # Calculate accuracy only on non-padding tokens
            correct_predictions = (predictions == shift_labels) & mask
            total_tokens = mask.sum()
            correct_tokens = correct_predictions.sum()

            # Gather the correct_tokens and total_tokens across all processes
            correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
            total_tokens = self.accelerator.gather_for_metrics(total_tokens)

            # Compute the mean token accuracy and log it
            total_sum = total_tokens.sum()
            accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
            self._metrics[mode]["mean_token_accuracy"].append(accuracy)

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "eval" if self.control.should_evaluate else "train"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items() if val}

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics[mode].clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="SFT",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
