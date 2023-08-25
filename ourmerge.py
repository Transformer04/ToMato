class OURAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0., locality_strength=None, use_local_init=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, sim):
        def visit_all_recursive(x, attn, N, C, sim):
            final_tokens = np.array([])
            visited = set()
            for index in range(N):
                if index not in visited:
                    result = [0] * C
                    num_merged = 0
                    visit_recursive(x, attn, index, visited, result, num_merged, sim)
                    final_tokens = np.concatenate((final_tokens, result(1, -1)), axis=0)

            return final_tokens


        def visit_recursive(x, attn, index, visited, result, num_merged, sim):
            if index not in visited:
                print(index, end=" ")
                visited.add(index)
                result += [x + y for x, y in zip(result, x[index])]
                num_merged += 1
                if index%14!=0 and attn[index][index-1] >= sim:
                    visit_recursive(x, attn, index-1, visited, sim)
                if index%14!=13 and attn[index][index+1] >= sim:
                    visit_recursive(x, attn, index+1, visited, sim)
                if index>13 and attn[index][index-14] >= sim:
                    visit_recursive(x, attn, index-14, visited, sim)
                if index<182 and attn[index][index+14] >= sim:
                    visit_recursive(x, attn, index+14, visited, sim)
    

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        nvtx.range_push("QKV linear") 
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        nvtx.range_pop() 

        nvtx.range_push("attn score")
        attn = (q @ k.transpose(-2, -1))
        nvtx.range_pop()

        final_tokens = visit_all_recursive(x[0], attn[0], N, C, sim)

        return final_tokens