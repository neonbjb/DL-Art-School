import torch
import torch.nn.functional as F


class ZeroPadDictCollate():
    """
    Given a list of dictionary outputs with torch.Tensors from a Dataset, iterates through each one, finds the longest
    tensor, and zero pads all the other tensors together.
    """
    def collate_tensors(self, batch, key):
        result = []
        largest_dims = [0 for _ in range(len(batch[0][key].shape))]
        for elem in batch:
            result.append(elem[key])
            largest_dims = [max(current_largest, new_consideration) for current_largest, new_consideration in zip(largest_dims, elem[key].shape)]
        # Now pad each tensor by the largest dimension.
        for i in range(len(result)):
            padding_tuple = ()
            for d in range(len(largest_dims)):
                padding_needed = largest_dims[d] - result[i].shape[d]
                assert padding_needed >= 0
                padding_tuple = (0, padding_needed) + padding_tuple
            result[i] = F.pad(result[i], padding_tuple)

        return torch.stack(result, dim=0)


    def collate_into_list(self, batch, key):
        result = []
        for elem in batch:
            result.append(elem[key])
        return result

    def __call__(self, batch):
        first_dict = batch[0]
        collated = {}
        for key in first_dict.keys():
            if isinstance(first_dict[key], torch.Tensor):
                if len(first_dict[key].shape) > 0:
                    collated[key] = self.collate_tensors(batch, key)
                else:
                    collated[key] = torch.stack([b[key] for b in batch])
            else:
                collated[key] = self.collate_into_list(batch, key)
        return collated