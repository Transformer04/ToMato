import torch

def custom_scatter_reduce(dim, dst, idx, src, reduce='mean'):

    dst.scatter_(dim, idx, src)

    if reduce == 'mean':
        count = torch.ones_like(src)
        dst_count = dst.new_zeros(dst.shape)
        dst_count.scatter_add_(dim, idx, count)
        dst /= dst_count

    return dst

if __name__ == "__main__":
    custom_scatter_reduce()
