import numpy as np
import torch

from .base_convergent_loss import Base_Convergent_Loss


class Classification_Loss(Base_Convergent_Loss):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.regression = False
        self.loss_weight = self.get_cfgs('loss_weight')
        if isinstance(self.loss_weight, np.ndarray):
            self.loss_weight = torch.from_numpy(self.loss_weight)
        self.ignore_index = self.get_cfgs('ignore_index')
        self.jitter_pool = self.get_cfgs('jitter_pool', default='mean')
        self.view_pool = self.get_cfgs('view_pool', default='max')
        self.signal_to_noise_ratio = self.get_cfgs('signal_to_noise_ratio', default=.9)
        self.to_each_view_its_own_label = self.get_cfgs('to_each_view_its_own_label')
        self.output_shape = self.get_cfgs('output_shape')

        assert(self.signal_to_noise_ratio > .5 and
               self.signal_to_noise_ratio < 1), \
            'signal to noise ration should be more than .5 and less than 1'

        assert(self.view_pool.lower() in ['max', 'mean']), \
            'Unknown view pool "%s"' % (self.view_pool)
        assert(self.jitter_pool.lower() in ['max', 'mean']), \
            'Unknown jitter pool "%s"' % (self.jitter_pool)

    def pool_and_reshape_output(self, output, num_views):
        output = output.view([-1, *self.output_shape])
        output = output.permute([1, 0, 2])
        # At this point, the output is in the shape of:
        # [num_jitter x batch_size x num_classes]

        x = np.cumsum([0, *num_views])
        if not self.to_each_view_its_own_label:
            if self.view_pool.lower() == 'mean':
                # The max() returns a tuple and hence the [0] for selecting not the index but the actual value
                vals = [output[:, x[i]:x[i + 1], :].mean(dim=1, keepdim=True) for i in range(len(x) - 1)]
            elif self.view_pool.lower() == 'max':
                vals = [output[:, x[i]:x[i + 1], :].max(dim=1, keepdim=True)[0] for i in range(len(x) - 1)]
            else:
                raise NameError(
                    f'There is no implementation of the view pool {self.view_pool} (see {self.modality.get_name()})')

            # Concatenate the values into the output along the first dimension
            output = torch.cat(vals, dim=1)

        pi_output = output.clone()

        if self.jitter_pool.lower() == 'mean'.lower():
            output = output.mean(dim=0)
        elif self.jitter_pool.lower() == 'max'.lower():
            output, _ = output.max(dim=0)
        # Alight, now it's [batch_size x num_classes]

        return output, pi_output

    def pool_and_reshape_target(self, target):
        # TODO: Ali: This seems like a bad idea - it basically drops all the structure...
        return target.view(-1)
