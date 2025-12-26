import torch


def topk_compress(gradients, k_ratio=0.1):
    """
    Apply top-k gradient compression.

    Args:
        gradients (List[Tensor]): model gradients
        k_ratio (float): fraction of gradients to keep

    Returns:
        List[Tensor]: compressed gradients
    """
    compressed = []

    for g in gradients:
        if g is None:
            compressed.append(None)
            continue

        flat = g.view(-1)
        k = max(1, int(k_ratio * flat.numel()))

        _, indices = torch.topk(flat.abs(), k)
        mask = torch.zeros_like(flat)
        mask[indices] = 1.0

        compressed_flat = flat * mask
        compressed.append(compressed_flat.view_as(g))

    return compressed
