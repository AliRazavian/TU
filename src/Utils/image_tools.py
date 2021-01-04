import os
import numpy as np
import math


backend = os.environ.get('IMAGE_BACKEND', 'cv2').lower()
if backend == 'cv2':
    import cv2
elif backend == 'skimage':
    from skimage import io
    from skimage.transform import warp, AffineTransform, resize


def get_random_transformed_image(
        im,
        output_size,
        augmentation_index,
):
    input_size = np.array(im.shape)
    M = get_random_spatial_transform_matrix(input_size, output_size, augmentation_index)
    if backend == 'cv2':
        tr_im = cv2.warpPerspective(im, M, (int(output_size[1]), int(output_size[0])))
    if backend == 'skimage':
        tform = AffineTransform(M)
        tr_im = warp(im, tform.inverse, output_shape=(int(output_size[0]), int(output_size[1])))

    if tr_im.ndim == 2:
        tr_im = tr_im.reshape([*tr_im.shape, 1])

    return tr_im, M, input_size


def get_fixed_transformed_image(
        im,
        output_size,
        augmentation_index,
):
    input_size = np.array(im.shape)
    M = get_fixed_spatial_transform_matrix(input_size, output_size, augmentation_index)
    if backend == 'cv2':
        tr_im = cv2.warpPerspective(im, M, (int(output_size[1]), int(output_size[0])))
    if backend == 'skimage':
        tform = AffineTransform(M)
        tr_im = warp(im, tform.inverse, output_shape=(int(output_size[0]), int(output_size[1])))
    if tr_im.ndim == 2:
        tr_im = tr_im.reshape([*tr_im.shape, 1])
    return tr_im, M, input_size


def get_inverse_transformed_image(
        tr_im,
        input_size,
        M,
):
    if backend == 'cv2':
        im = cv2.warpPerspective(tr_im, M, (int(input_size[1]), int(input_size[0])))
    if backend == 'skimage':
        tform = AffineTransform(np.linalg.inv(M))
        im = warp(tr_im, tform.inverse, output_shape=(int(input_size[0]), int(input_size[1])))
    return im


def get_random_spatial_transform_matrix(
        input_size,
        output_size,
        augmentation_index=0,
):
    # With deep learning, data augmentation increases the accuracy of the model,
    # which is why we like to have as few copies of the same sample as possible
    # during the traning process. Hence we ignore the augmentation_index

    # If -in the future- it is discovered that we should augment the data
    # during the traning, we can use augmentation_index accordingly.
    rotation = [0, 90][np.random.randint(2)]
    if np.random.rand() > .5:
        rotation += np.random.normal(0, 15)
    return get_cropped_patch(input_size=input_size,
                             output_size=output_size,
                             center_ratio=np.random.normal(0, .2, 2),
                             rotation=rotation,
                             stretch=np.random.normal(1, .1, 2),
                             # Mirror horizontally slightly less than random as we want
                             # the system to know left-right from the direction
                             horizontal_mirror=np.random.rand() > .6,
                             vertical_mirror=np.random.rand() > .5)


def get_fixed_spatial_transform_matrix(input_size, output_size, augmentation_index=0):
    fix_param = get_fix_spatial_transform_params(augmentation_index)
    return get_cropped_patch(input_size=input_size,
                             output_size=output_size,
                             center_ratio=fix_param['center_ratio'],
                             rotation=fix_param['rotation'],
                             stretch=fix_param['stretch'],
                             horizontal_mirror=fix_param['horizontal_mirror'],
                             vertical_mirror=fix_param['vertical_mirror'])


def get_fix_spatial_transform_params(augmentation_index):
    # good number for augmentations are :
    # 1: the image itself
    # 4: images flipped horizontally and vertically
    # 12: the 4 images are rotated -10, 0 and 10 degrees
    # 108: corner cropped of those 12 images
    center_ratio = [0, -1 / 4, 1 / 4]
    rotation = [0, -10, 10]
    horizontal_mirror = [False, True]
    vertical_mirror = [False, True]
    cnt = 0
    while True:
        for c_x in center_ratio:
            for c_y in center_ratio:
                for r in rotation:
                    for h in horizontal_mirror:
                        for v in vertical_mirror:
                            cnt += 1
                            if augmentation_index < cnt:
                                return {
                                    'center_ratio': (c_x, c_y),
                                    'stretch': (1 - abs(c_x), 1 - abs(c_y)),
                                    'rotation': r,
                                    'horizontal_mirror': h,
                                    'vertical_mirror': v
                                }
        # if we reach to this stage, it means our augmentation size has been
        # bigger than 108, so we make the cropped images smaller and the angle of rotations more:
        rotation = [r * 1.5 for r in rotation]
        center_ratio = [c * .1 for c in center_ratio]
        # and we go back to the "while True" loop to sample another 108 images.
        # Honestly, I don't think we will ever need to augment and image more
        # than 12 times but I just add support for infinite data augmentation


def get_cropped_patch(
        input_size,
        output_size,
        center_ratio=(0, 0),
        rotation=0,
        stretch=(1, 1),
        horizontal_mirror=False,
        vertical_mirror=False,
):

    input_corners = np.array([[0, 0, 1], [input_size[1], 0, 1], [input_size[1], input_size[0], 1], [0, input_size[0], 1]],
                             dtype='float32').T
    output_corners = np.array([[0, 0, 1], [output_size[1], 0, 1], [output_size[1], output_size[0], 1], [0, output_size[0], 1]],
                              dtype='float32').T

    input_center = np.mean(input_corners, axis=1)
    output_center_in_original_image = (np.array([*center_ratio, 0]) + 1) * input_center
    output_center_in_cropped_image = np.mean(output_corners, axis=1)

    output_corners_in_original_image = output_corners.copy()
    output_corners_in_original_image -= output_center_in_cropped_image[:, np.newaxis]
    output_corners_in_original_image *= np.array([*stretch, 0])[:, np.newaxis]
    output_corners_in_original_image[2, :] = 1
    output_corners_in_original_image = np.dot(rotate(rotation), output_corners_in_original_image)
    output_corners_in_original_image += output_center_in_original_image[:, np.newaxis]
    output_corners_in_original_image[2, :] = 1

    if (horizontal_mirror):
        output_corners = output_corners[:, [1, 0, 3, 2]]
    if (vertical_mirror):
        output_corners = output_corners[:, [3, 2, 1, 0]]

    # numpy can solve AX=B, here we want to solve M . original = cropped
    # To do so, we must reformulate the equation:
    # (M . original)' = cropped' # => original' . M' = cropped'

    M = np.linalg.lstsq(output_corners_in_original_image.T, output_corners.T, rcond=-1)[0].T
    M[np.abs(M) < 1e-4] = 0  # because it is visually more appealing

    return M


def convert_original_points_to_cropped_image(M,
                                             original_points,  # must be a numpy array of nx2 coordinates in relative values
                                             original_im_size: (int, int),
                                             cropped_im_size: (int, int)):

    assert(original_points.max() <= 1), f'point values must be between 0 to 1 but max value is {original_points.max()}'
    assert(original_points.min() >= 0), f'point values must be between 0 to 1 but min value is {original_points.min()}'

    extra_ones = np.ones((original_points.shape[0], 1))
    original_points = np.concatenate((original_points, extra_ones), axis=1)

    original_points[:, 0] *= original_im_size[1]
    original_points[:, 1] *= original_im_size[0]

    tr_points = np.dot(M, original_points.T).T
    tr_points /= np.expand_dims(tr_points[:, 2], 1)
    tr_points = tr_points[:, 0:2]

    tr_points[:, 0] /= cropped_im_size[1]
    tr_points[:, 1] /= cropped_im_size[0]

    return tr_points


def convert_cropped_image_points_to_original(M,
                                             tr_points,  # must be a numpy array of nx2 coordinates in relative values
                                             original_im_size: (int, int),
                                             cropped_im_size: (int, int)):

    extra_ones = np.ones((tr_points.shape[0], 1))
    tr_points = np.concatenate((tr_points, extra_ones), axis=1)

    tr_points[:, 0] *= cropped_im_size[1]
    tr_points[:, 1] *= cropped_im_size[0]

    original_points = np.dot(np.linalg.inv(M), tr_points.T).T
    original_points /= np.expand_dims(original_points[:, 2], 1)
    original_points = original_points[:, 0:2]

    original_points[:, 0] /= original_im_size[1]
    original_points[:, 1] /= original_im_size[0]

    return original_points


# I'm only using the rotate function at the moment, the rest are implemented inside
# get_cropped_patch. But I leave the functions here in case we later on decide to
# test something


def rotate(
        theta,
        M=np.eye(3, dtype='float32'),
):
    theta = theta * math.pi / 180
    rt = np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]],
        dtype='float32',
    )
    rot_M = rt
    return np.dot(rot_M, M)


def streach(
        rho_x,
        rho_y,
        M=np.eye(3, dtype='float32'),
):
    sc_M = np.array([[rho_x, 0, 0], [0, rho_y, 0], [0, 0, 1]], dtype='float32')
    return np.dot(sc_M, M)


def translate(
        x,
        y,
        M=np.eye(3, dtype='float32'),
):
    tr_M = np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], dtype='float32')
    return np.dot(tr_M, M)


def mirror(M=np.eye(3, dtype='float32')):
    tr_M = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype='float32')
    return np.dot(tr_M, M)


# all images should be float, with the dimension: w x h x c
# where either w or h or both are equal to the "scale_to" config


def fix_dimension_and_normalize(
        im,
        keep_aspect,
        scale_to,
        colorspace,
):
    # add extra dimension if it doesn't have any
    if im.ndim == 2:
        im = im.reshape([*im.shape, 1])

    original_im_size = np.array(im.shape)

    # First, we resize the image
    if (keep_aspect):
        scale_size = (scale_to, scale_to)
    else:
        min_shape = min(im.shape[0], im.shape[1])
        scale_size = (int(scale_to * im.shape[0] / min_shape), int(scale_to * im.shape[1] / min_shape))

    if backend == 'cv2':
        im = cv2.resize(im, scale_size, interpolation=cv2.INTER_LINEAR)
    if backend == 'skimage':
        im = resize(im, scale_size, order=1, mode='constant', anti_aliasing=False)

    # Then we convert the image so that the values be in the range of [0-1]
    if im.dtype in ['uint16', 'uint8', 'int16', 'int8']:
        max_val = np.iinfo(im.dtype).max
        min_val = np.iinfo(im.dtype).min
    elif im.dtype in ['float32', 'float64']:
        assert(im.max() <= 1. and im.min() >= .0),\
            "after resizing the image with skimage, the max and min are %f,%f" % (
                im.max(), im.min())
        max_val = 1
        min_val = 0
    else:
        raise BaseException('Unknown image format type: "%s"' % (im.dtype))

    im = im.astype('float32')
    im -= min_val
    im /= (max_val - min_val)

    # removing the alpha channel:
    if im.shape[2] == 4:
        im = im[:, :, :3]

    # fix colorspace:
    if im.shape[2] == 1 and colorspace.lower() == 'RGB'.lower():
        im = np.repeat(im, 3, axis=2)
    elif im.shape[2] == 3 and colorspace.lower() == 'Gray'.lower():
        im = 0.2989 * im[:, :, 0] + 0.5870 * im[:, :, 1] + 0.1140 * im[:, :, 2]
        im = im.reshape([*im.shape, 1])
    return im, original_im_size


def load_image(im_path):
    success = False

    backend = os.environ.get('IMAGE_BACKEND', 'cv2')

    if backend.lower() == 'cv2':
        im = cv2.imread(im_path)
        if im is None:
            im = np.zeros((10, 10, 3), dtype='uint8')
        else:
            success = True

    elif backend.lower() == 'skimage':
        try:
            im = io.imread(im_path)
            success = True
        except FileNotFoundError:
            im = np.zeros((10, 10, 3), dtype='uint8')

    else:
        raise Exception('No image backend matching %s'.format(backend))

    return im, success


def plot_points_on_image(im, points):
    import matplotlib.pyplot as plt
    w = im.shape[0]
    h = im.shape[1]

    print(im.shape)

    print(im.max(), im.min())

    x1 = int(points[0, 0] * h)
    y1 = int(points[0, 1] * w)
    x2 = int(points[1, 0] * h)
    y2 = int(points[1, 1] * w)

    print(x1, y1, '->', x2, y2)
    cv2.line(im, (x1, y1), (x2, y2), (255, 255, 0), 1)
    plt.imshow(im)
    plt.show()


if __name__ == "__main__":
    im, _ = load_image('test_data/imgs/Wrist/1st_export/Export 2006 to 2009/10072--23924--2009-10-13-09.04.19--left--Handledfrontalsin.png')
    points = np.array([[0.69, 0.78],
                       [0.68, 0.56]])
    plot_points_on_image(im.copy(), points)

    for j in range(4):
        tr_im, M, input_size = get_random_transformed_image(im, (256, 256), j)
        tr_points = convert_original_points_to_cropped_image(M,
                                                             points,
                                                             original_im_size=(im.shape[0], im.shape[1]),
                                                             cropped_im_size=(256, 256))
        plot_points_on_image(tr_im.copy(), tr_points)

        back_points = convert_cropped_image_points_to_original(M,
                                                               tr_points,
                                                               original_im_size=(im.shape[0], im.shape[1]),
                                                               cropped_im_size=(256, 256))
        plot_points_on_image(im.copy(), back_points)
