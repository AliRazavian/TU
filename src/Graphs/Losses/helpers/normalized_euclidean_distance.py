import torch


def normalized_euclidean_distance(output):
    epsilon = 1e-12
    norm = output.norm(p=2, dim=1, keepdim=True) + epsilon
    norm_output = output.div(norm)
    # Original method - before the cdist in PyTorch 1.1
    # return torch.sqrt(2 - 2 * torch.mm(norm_output, norm_output.t()).clamp(max=1, min=-1) + epsilon)
    return torch.cdist(norm_output, norm_output)
