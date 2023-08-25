# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------

from typing import Tuple

import torch
from models import Attention, Block, VisionTransformer

from tome.merge import ToMato
from tome.utils import parse_r

import torch.cuda.nvtx as nvtx

class ToMeBlock(Block):

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        nvtx.range_push("norm")
        norm = self.norm1(x)
        nvtx.range_pop()
        nvtx.range_push("attention")
        x_attn, metric = self.attn(norm, attn_size)
        nvtx.range_pop()

        nvtx.range_push("drop_path")
        x = x + self._drop_path1(x_attn)
        nvtx.range_pop()

        r = self._tome_info["r"]
        print("r")
        print(r)
       
        print(x.data.shape)

        if self._tome_info["ToMato"] < 1:
            if r > 0:
                nvtx.range_push("ToME")
                # Apply ToMe here
                tm = ToMato()
                x, self._tome_info["size"] = tm.tomato(
                    x,
                    r,
                    self._tome_info["class_token"],
                    self._tome_info["distill_token"],
                )
                nvtx.range_pop()

        self._tome_info["ToMato"] += 1        

        nvtx.range_push("norm")
        norm2 = self.norm2(x)
        nvtx.range_pop()
        nvtx.range_push("feed forward")
        mlp = self.mlp(norm2)
        nvtx.range_pop()
        nvtx.range_push("drop_path")
        x = x + self.drop_path(mlp)
        nvtx.range_pop()

        #x = x + self._drop_path2(self.mlp(self.norm2(x)))
        return x


class ToMeAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        nvtx.range_push("QKV linear")
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)
        nvtx.range_pop()

        nvtx.range_push("attn score")
        attn = (q @ k.transpose(-2, -1)) * self.scale
        nvtx.range_pop()

        # Apply proportional attention
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        
        nvtx.range_push("softmax")
        attn = attn.softmax(dim=-1)
        nvtx.range_pop()
        nvtx.range_push("attn drop")
        attn = self.attn_drop(attn)
        nvtx.range_pop()

        nvtx.range_push("attn value")
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        nvtx.range_pop()

        nvtx.range_push("proj")
        x = self.proj(x)
        x = self.proj_drop(x)
        nvtx.range_pop()

        # Return k as well here
        return x, k.mean(1)


def make_tome_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._tome_info["r"] = self.r
            self._tome_info["size"] = None
            self._tome_info["source"] = None
            self._tome_info["ToMato"] = 0

            return super().forward(*args, **kwdargs)

    return ToMeVisionTransformer


def apply_patch(
    model: VisionTransformer, prop_attn: bool = True
):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    ToMeVisionTransformer = make_tome_class(model.__class__)

    model.__class__ = ToMeVisionTransformer
    model.r = 0
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None,
        "distill_token": False,
        "ToMato":0,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = ToMeBlock
            module._tome_info = model._tome_info
        elif isinstance(module, Attention):
            module.__class__ = ToMeAttention
