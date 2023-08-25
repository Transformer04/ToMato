# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
import sys
import os
sys.path.append(os.path.abspath("/home/smh-ewha/OURPROJ/ToMato/tome"))

from typing import Callable, Tuple

import torch
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx

def do_nothing(x, mode=None):
    return x

class ToMato():
    def __init__(self):
        self.result = torch.Tensor([0]*768)
        self.num_merged = 0

    def tomato(
        self,
        x: torch.Tensor,
        r: float,
        class_token: bool = False,
        distill_token: bool = False,
    ) -> Tuple[Callable, Callable]:

        if r <= 0:
            return do_nothing, do_nothing
        
        print("r")
        print(r)
        
        with torch.no_grad():
            B, N, C = x.shape
            metric = x[0]

            if class_token:
                cls_token = metric[:1]
                metric = metric[1:]
            if distill_token:
                dis_token = metric[-1:]
                metric = metric[:-1]

            attn = metric @ metric.transpose(-2, -1)

            flatten_attn = attn.view(-1)
            normalized_attn = F.softmax(flatten_attn, dim=0)
            
            attn = normalized_attn.view(attn.shape)

        def visit_all_recursive(x, attn, N, C, sim):
            final_tokens = torch.Tensor([[0]*C])
            size = torch.Tensor([[0]])
            visited = set()
            for index in range(N-1):
                if index not in visited:
                    self.result = torch.Tensor([0]*C)
                    self.num_merged = 0
                    visit_recursive(x, attn, index, visited, sim)
                    self.result = self.result / self.num_merged
                    final_tokens = torch.cat((final_tokens, self.result.unsqueeze(0)), dim=0)
                    size = torch.cat((size, torch.tensor([[self.num_merged]])), dim=0)
            
            final_tokens = final_tokens[1:]
            size = size[1:]
            return final_tokens, size

        def visit_recursive(x, attn, index, visited, sim):
            if index not in visited:
                visited.add(index)
                self.result = self.result + x[index]
                self.num_merged += 1
                if index%14!=0 and attn[index][index-1] >= sim:
                    visit_recursive(x, attn, index-1, visited, sim)
                if index%14!=13 and attn[index][index+1] >= sim:
                    visit_recursive(x, attn, index+1, visited,  sim)
                if index>13 and attn[index][index-14] >= sim:
                    visit_recursive(x, attn, index-14, visited, sim)
                if index<182 and attn[index][index+14] >= sim:
                    visit_recursive(x, attn, index+14, visited, sim)

        final_tokens, size = visit_all_recursive(metric, attn, N, C, r)
        if distill_token:
            out = torch.cat((cls_token, final_tokens, dis_token), dim=0).unsqueeze(0)
            size = torch.cat((torch.Tensor([[1]]), size, torch.Tensor([[1]])), dim=0).unsqueeze(0)
        else:
            out = torch.cat((cls_token, final_tokens), dim=0).unsqueeze(0)
            size = torch.cat((torch.Tensor([[1]]), size), dim=0).unsqueeze(0)


        return out, size


def merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax")
    return source
