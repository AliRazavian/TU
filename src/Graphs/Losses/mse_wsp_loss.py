import numpy as np
import torch

from .base_convergent_loss import Base_Convergent_Loss
import Utils.image_tools as image_tools


class MSE_WSP_Loss(Base_Convergent_Loss):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pi_model = False
        self.classification = False
        self.reconstruction = False
        self.regression = self.get_cfgs('regression')
        self.loss_weight = self.get_cfgs('loss_weight')
        self.ignore_index = self.get_cfgs('ignore_index')
        self.jitter_pool = self.get_cfgs('jitter_pool', default='mean')
        self.view_pool = self.get_cfgs('view_pool', default='max')
        self.to_each_view_its_own_label = True
        self.output_shape = self.get_cfgs('output_shape')

    def pool_and_reshape_output(self, output, num_views):
        # we want the values to be between 0 to 1
        return output

    def pool_and_reshape_target(self, target):
        return target

    def calculate_loss(self, output, target):
        return 0

    def calculate_regression_loss(self, batch):
        # Alright, here we go

        output = torch.sigmoid(batch[self.output_name])

        target = batch[self.target_name]
        spatial_transform = batch['spatial_transforms']
        original_input_size = batch['input_size']
        images = batch['encoder_image']

        output = output.view(target.shape)
        origianl_predictions = np.ndarray(list(target.shape), dtype=float)
        loss = 0

        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                im = images[i, j, :, :, :]
                original_target = target[i, j, :, :].data.cpu().numpy()
                M = spatial_transform[i, j, :, :].data.cpu().numpy()
                sz = original_input_size[i, j, :].data.cpu().numpy()

                origianl_predictions[i, j, :, :] = image_tools.convert_cropped_image_points_to_original(
                    M, output[i, j, :, :].data.cpu().numpy(), (sz[1], sz[0]), (im.shape[-2], im.shape[-1]))

                if np.isnan(original_target).any():
                    continue

                cropped_target = image_tools.convert_original_points_to_cropped_image(
                    M, original_target, (sz[1], sz[0]), (im.shape[-2], im.shape[-1]))

                # below lines visualizes the ground truth on the images
                # image_tools.plot_points_on_image(im.squeeze().data.numpy() * 255,
                #                                 cropped_target)

                diff = output[i, j, :, :] - torch.FloatTensor(cropped_target).to(output.device)
                loss += torch.sqrt((diff**2).sum())

        batch['predicted_' + self.output_name] = origianl_predictions
        return loss / target.shape[0] / target.shape[1]
