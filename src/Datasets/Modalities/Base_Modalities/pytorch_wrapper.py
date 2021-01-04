import os
import re
import numpy as np
import torch
from typing import List
from torch._six import string_classes, int_classes, container_abcs


def convert_2_cuda(data):
    if not torch.is_tensor(data):
        return data

    if ('DEVICE_BACKEND' in os.environ and os.environ['DEVICE_BACKEND'].lower() in ['cuda', 'gpu']):
        return data.cuda()

    return data


def wrap(data):
    assert(isinstance(data, np.ndarray)),\
        'unknown modality data type %s' % (type(data))
    data_type = data.dtype.name
    if data_type.startswith('object'):
        # TODO: Check when this stack occurs as it will most likely not be handled by collate
        data = np.stack(data)
        data_type = data.dtype.name

    if data_type.startswith('float'):
        data = torch.FloatTensor(data)
    elif data_type.startswith('int'):
        data = torch.LongTensor(data)

    else:
        raise BaseException('Unknown numpy data type: "%s"' % (data_type))

    data = convert_2_cuda(data)
    return data


def unwrap(data):
    # Already in numpy format
    if isinstance(data, np.ndarray):
        return data

    assert(isinstance(data, torch.Tensor)),\
        'unknown modality data type %s' % (type(data))

    data = data.detach().clone()

    # if os.environ['DEVICE_BACKEND'].lower() in ['cuda'.lower(), 'gpu'.lower()]:
    if data.is_cuda:
        data = data.cpu()
    data = data.numpy()
    return data


_use_shared_memory = False
r"""Whether to use shared memory in collate"""

numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


def collate_factory(keys_2_ignore: List[str]):
    r"""
    Returns a function for use with DataLoader that:
    1. puts each data field into a tensor with outer dimension batch size
    2. moves tensor to cuda if DEVICE_BACKEND is CUDA
    3. ignores any dictionary keys specified in the list

    Based on PyTorch's collate function with a minor tweak to dict input
    """

    def collate(batch):
        r"""Puts each data field into a tensor with outer dimension batch size"""

        error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
        elem_type = type(batch[0])
        if isinstance(batch[0], torch.Tensor):
            out = None
            if _use_shared_memory:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = batch[0].storage()._new_shared(numel)
                out = batch[0].new(storage)
            return torch.stack(batch, 0, out=out)
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            elem = batch[0]
            if elem_type.__name__ == 'ndarray':
                # array of string classes and object
                if re.search('[SaUO]', elem.dtype.str) is not None:
                    raise TypeError(error_msg.format(elem.dtype))

                return torch.stack([torch.from_numpy(b) for b in batch], 0)
            if elem.shape == ():  # scalars
                py_type = float if elem.dtype.name.startswith('float') else int
                return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
        elif isinstance(batch[0], int_classes):
            return torch.LongTensor(batch)
        elif isinstance(batch[0], float):
            return torch.DoubleTensor(batch)
        elif isinstance(batch[0], string_classes):
            return batch
        elif isinstance(batch[0], container_abcs.Mapping):
            ret = {}
            if len(batch) != 1:
                raise ValueError('Expected a mapping to be of length 1 - this is a deviation from the original format')
            for key in batch[0]:
                if key in keys_2_ignore:
                    ret[key] = batch[0][key]
                else:
                    # Recursive collect
                    ret[key] = collate([d[key] for d in batch])
                    ret[key] = ret[key][0]
            return ret
        elif isinstance(batch[0], container_abcs.Sequence):
            transposed = zip(*batch)
            return [collate(samples) for samples in transposed]

        raise TypeError((error_msg.format(type(batch[0]))))

    return collate
