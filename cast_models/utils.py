"""Utilility function for all.
"""
# This code is borrowed and re-implemented from:
# https://github.com/jyhjinghwang/SegSort/blob/master/network/segsort/vis_utils.py
# https://github.com/jyhjinghwang/SegSort/blob/master/network/segsort/common_utils.py

import torch
import torch.nn.functional as F


def resize_labels(labels, size):
    """Helper function to resize labels.

    Args:
        labels: A long tensor of shape `[batch_size, height, width]`.

    Returns:
        A long tensor of shape `[batch_size, new_height, new_width]`.
    """
    n, h, w = labels.shape
    labels = F.interpolate(labels.view(n, 1, h, w).float(),
                           size=size,
                           mode='nearest')
    labels = labels.squeeze_(1).long()

    return labels


def calculate_principal_components(embeddings, num_components=3):
    """Calculates the principal components given the embedding features.

    Args:
      embeddings: A 2-D float tensor of shape `[num_pixels, embedding_dims]`.
      num_components: An integer indicates the number of principal
        components to return.

    Returns:
      A 2-D float tensor of shape `[num_pixels, num_components]`.
    """
    embeddings = embeddings - torch.mean(embeddings, 0, keepdim=True)
    _, _, v = torch.svd(embeddings)
    return v[:, :num_components]


def pca(embeddings, num_components=3, principal_components=None):
    """Conducts principal component analysis on the embedding features.

    This function is used to reduce the dimensionality of the embedding.

    Args:
        embeddings: An N-D float tensor with shape with the 
            last dimension as `embedding_dim`.
        num_components: The number of principal components.
        principal_components: A 2-D float tensor used to convert the
            embedding features to PCA'ed space, also known as the U matrix
            from SVD. If not given, this function will calculate the
            principal_components given inputs.

    Returns:
        A N-D float tensor with the last dimension as  `num_components`.
    """
    shape = embeddings.shape
    embeddings = embeddings.view(-1, shape[-1])

    if principal_components is None:
        principal_components = calculate_principal_components(
            embeddings, num_components)
    embeddings = torch.mm(embeddings, principal_components)

    new_shape = list(shape[:-1]) + [num_components]
    embeddings = embeddings.view(new_shape)

    return embeddings


def one_hot(labels, max_label=None):
    """Transform long labels into one-hot format.

    Args:
        labels: An N-D long tensor.

    Returns:
        An (N+1)-D long tensor.
    """
    if max_label is None:
        max_label = labels.max() + 1

    shape = labels.shape
    labels = labels.view(-1, 1)
    one_hot_labels = torch.zeros((labels.shape[0], max_label),
                                 dtype=torch.long,
                                 device=labels.device)
    one_hot_labels = one_hot_labels.scatter_(1, labels, 1)

    new_shape = list(shape) + [max_label]
    one_hot_labels = one_hot_labels.view(new_shape)

    return one_hot_labels


def normalize_embedding(embeddings, eps=1e-12):
    """Normalizes embedding by L2 norm.

    This function is used to normalize embedding so that the
    embedding features lie on a unit hypersphere.

    Args:
      embeddings: An N-D float tensor with feature embedding in
        the last dimension.

    Returns:
      An N-D float tensor with the same shape as input embedding
      with feature embedding normalized by L2 norm in the last
      dimension.
    """
    norm = torch.norm(embeddings, dim=-1, keepdim=True)
    norm = torch.where(torch.ge(norm, eps),
                       norm,
                       torch.ones_like(norm).mul_(eps))
    return embeddings / norm


def segment_mean(x, index):
    """Function as tf.segment_mean.
    """
    x = x.view(-1, x.shape[-1])
    index = index.view(-1)

    max_index = index.max() + 1
    sum_x = torch.zeros((max_index, x.shape[-1]),
                        dtype=x.dtype,
                        device=x.device)
    num_index = torch.zeros((max_index,),
                            dtype=x.dtype,
                            device=x.device)

    num_index = num_index.scatter_add_(
        0, index, torch.ones_like(index, dtype=x.dtype))
    num_index = torch.where(torch.eq(num_index, 0),
                            torch.ones_like(num_index, dtype=x.dtype),
                            num_index)

    index_2d = index.view(-1, 1).expand(-1, x.shape[-1])
    sum_x = sum_x.scatter_add_(0, index_2d, x)
    mean_x = sum_x.div_(num_index.view(-1, 1))

    return mean_x


def segment_mean_nd(x, index, start_dim=-3, end_dim=-2):
    """Function segment_mean for nd-array as inputs.

    For the inputs, the last dimension denotes feature channels,
    which are averaged according to label indices.  The label
    index is shared among `start_dim` and `end_dim`, but independent
    from the rest of dimensions.

    For example, if input is a 4-D tensor with `start_dim` and `end_dim`
    as -3 and -2. It means that the input is a 2-D label map and the
    first dimension indicates batch.

    Args:
        x: A `float` tensor of shape `[D(0), ..., D(N-1)]`.
        index: An `integer` tensor of shape `[D(0), ..., D(N-2)]`.
        start_dim: An `integer` indicates the starting dimension
            of label index.
        end_dim: An `integer` indicates the ending dimension
            of label index.
    Returns:
        mean_x: A `float` tensor of shape 
    """
    batch_sizes = list(x.shape[:start_dim])
    channel = x.shape[-1]

    if start_dim < 0:
        start_dim = len(x.shape) + start_dim
    if end_dim < 0:
        end_dim = len(x.shape) + end_dim

    # Flatten tensors.
    index = index.flatten(start_dim, end_dim).flatten(0, start_dim-1)
    x = x.flatten(start_dim, end_dim).flatten(0, start_dim-1)

    # Separate label index by batch.
    batch_inds = torch.arange(0, x.shape[0],
                              dtype=index.dtype, device=index.device)
    max_ind = index.max() + 1
    index = batch_inds.view(-1, 1) * max_ind + index
    mean_x = segment_mean(x.contiguous(), index)

    # If the minimum label index of the last batch is less than
    # `max_ind`, the averaged `x` would be shorter than expected.
    expected_n = (batch_inds.max() + 1) * max_ind
    if mean_x.shape[0] < expected_n:
        mean_x = F.pad(mean_x, (0, 0, 0, expected_n - mean_x.shape[0]),
                       'constant', 0)
    mean_x = mean_x.view(batch_sizes + [max_ind] + [channel])

    return mean_x
