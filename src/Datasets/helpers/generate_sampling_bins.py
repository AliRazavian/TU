import math
import numpy as np
import pandas as pd
import operator
from UIs.console_UI import Console_UI


def generate_sampling_bins(annotations, batch_size: int, silent: bool = True):
    assert 'num_views' in annotations, 'Expected num_views in annoations'
    assert isinstance(annotations.index, pd.MultiIndex), 'The annoation data must be a multi-index'

    total_samples = len(annotations)
    # We need to allow different batch sizes depending on the task/scenes complexity
    bin_counts = math.ceil(total_samples / batch_size / 2)

    bins = []

    # Do NOT replace this loop with []*bin_counts
    for _ in range(bin_counts):
        bins.append([])

    bin_weights = np.zeros(bin_counts, dtype='int')
    max_views = min(int(annotations['num_views'].max()), batch_size)
    for num_views in range(max_views, 0, -1):
        sub_df = annotations[annotations['num_views'] == num_views]
        # The codes contains the internal representation of the value and does not map
        # properly to the original index (i.e. max(internal_codes_representation) < max(annotations.index))
        internal_codes_representation = sub_df.index.codes[0]
        indices = list(set(internal_codes_representation))
        # - old code? prev line: sub_df.index.levels[0][internal_codes_representation]))
        random_indices = np.random.permutation(indices)

        while (len(random_indices) > 0):
            # Divising by a number bigger than batch_size will always result in the
            # all max(bin_caps) == 0 as bin_weights is strictly positive
            bin_caps = (batch_size - bin_weights) // max(min(num_views, batch_size - 1), 1)
            valid_bins = np.random.permutation(np.nonzero(bin_caps > 0)[0])
            if (len(valid_bins) == 0):  # adding new bins
                num_new_bins = math.ceil(len(random_indices) * num_views / batch_size)
                for _ in range(num_new_bins):
                    bins.append([])
                bin_weights = np.hstack([bin_weights, np.zeros(num_new_bins, dtype='int')])
                if not silent:
                    Console_UI().inform_user(
                        f'Generating sampling bins, couldn\'t fit everything in {len(bins) - num_new_bins} bins,' +
                        f'trying adding another {num_new_bins} bins. Total will be {len(bins)} bins')
                continue
            to_handle_count = min(len(valid_bins), len(random_indices))
            for i in range(to_handle_count):
                bins[valid_bins[i]].append(random_indices[i])

            random_indices = random_indices[to_handle_count:]
            bin_weights[valid_bins[:to_handle_count]] += num_views

    # We need to remove [] from the sampled bins
    valid_bin_idxs = [i for i in range(len(bins)) if len(bins[i]) > 0]
    bin_weights = [bin_weights[i] for i in valid_bin_idxs]
    bins = [bins[i] for i in valid_bin_idxs]

    # Sort so that the heaviest bins come first
    # this allows pytorch to allocate full memory from start
    # + reduces the risk of late crashes when full allocations suddenly appear
    sort_args = (bin_weights, [len(bin) for bin in bins])
    bins = [x for _, __, x in sorted(
        zip(*sort_args, bins),
        key=operator.itemgetter(0, 1),
    )][::-1]
    bin_weights = [x for x, _ in sorted(
        zip(*sort_args),
        key=operator.itemgetter(0, 1),
    )][::-1]
    return bins, bin_weights
