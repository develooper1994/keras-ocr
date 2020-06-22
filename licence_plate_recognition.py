import argparse
import os

import numpy as np
from PIL import Image

import keras_ocr

import torch
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image

from esrgan.models import GeneratorRRDB

try:
    from datasets import denormalize, mean, std
    from models import GeneratorRRDB
except:
    try:
        from esrgan.datasets import denormalize, mean, std
        from esrgan.models import GeneratorRRDB
    except:
        from implementations.esrgan.datasets import denormalize, mean, std
        from implementations.esrgan.models import GeneratorRRDB


def plot_and_save(images, prediction_groups):
    fig, axs = plt.subplots(nrows=len(images), figsize=(20, 20))
    # for ax, image, predictions in zip(axs, images, prediction_groups):
    #     keras_ocr.tools.drawAnnotations(image=image, predictions=predictions, ax=ax)
    keras_ocr.tools.drawAnnotations(image=images[0], predictions=prediction_groups[0], ax=axs)
    plt.show()
    fig.savefig("figure_" + image_name)


def sort_bounindg_boxes(prediction_groups):
    prediction_group = prediction_groups[0]
    arr_prediction_group = np.array(prediction_group)
    bounding_boxes = arr_prediction_group[:, 1]  # TODO! IndexError: too many indices for array
    bounding_boxes = np.stack(bounding_boxes)

    # %% only to detect multiple characters in once. each bounding box has 4 coordinate points.
    left_upper_y = bounding_boxes[:, 1, 1]
    shorted_ids_y = np.argsort(left_upper_y)
    # bounding_boxes = bounding_boxes[shorted_ids_y, :, :]
    sorted_arr_prediction_group = arr_prediction_group[shorted_ids_y]

    left_upper_x = bounding_boxes[:, 1, 0]
    shorted_ids_x = np.argsort(left_upper_x)
    # bounding_boxes = bounding_boxes[shorted_ids_x, :, :]
    sorted_arr_prediction_group = sorted_arr_prediction_group[shorted_ids_x]
    sorted_list_prediction_group = sorted_arr_prediction_group.tolist()
    sorted_list_prediction_group = [sorted_list_prediction_group]
    return sorted_list_prediction_group


class recognize:
    """
    Usage;
    recognizer = recognize()
    prediction_groups = recognizer(images)
    plate_string = recognizer.get_plate_string(prediction_groups)
    """
    def __init__(self, super_resolution_checkpoint_model=None, channels=None, residual_blocks=None):
        self.pipeline = keras_ocr.pipeline.Pipeline()

        # Super-resolution
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # channels -> Number of image channels
        if channels is None:
            channels = 3
        if residual_blocks is None:
            residual_blocks = 23
        self.generator = GeneratorRRDB(channels, filters=64, num_res_blocks=residual_blocks).to(self.device)
        if super_resolution_checkpoint_model is None:
            super_resolution_checkpoint_model = r"D:\PycharmProjects\ocr_toolkit\esrgan\saved_models\generator_151.pth"
        self.generator.load_state_dict(torch.load(super_resolution_checkpoint_model))  # Path to checkpoint model
        self.generator.eval()
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    def __call__(self, images):
        return self.recognize(images)

    def recognize(self, images):
        if not isinstance(images, np.ndarray):
            images = [image for image in images]
        prediction_groups = self.pipeline.recognize(images)
        prediction_groups = sort_bounindg_boxes(prediction_groups)
        # try:
        #     prediction_groups = sort_bounindg_boxes(prediction_groups)
        # except:
        #     None
        return prediction_groups

    def get_plate_string(self, predictions, pad_char=''):
        prediction_array = np.array(predictions[0])
        prediction_array_strings = prediction_array[:, 0]
        prediction_strings_list = prediction_array_strings.tolist()
        plate_string = pad_char.join(prediction_strings_list)
        # plate_string = pad_char.join(prediction_strings_list[1:])  # removes 'tr'
        return plate_string

    def superresolution(self, images):
        if not isinstance(images, np.ndarray):
            images = [image for image in images]
        sr_images = []
        for image in images:
            image_tensor = self.transform(image).to(self.device).unsqueeze(0)
            with torch.no_grad():
                enhanced = self.generator(image_tensor)
                sr_image = denormalize(enhanced)
                ndarr = sr_image.mul(255).add_(0.5).clamp_(0, 255)[0].permute(1, 2, 0).detach().to('cpu', torch.uint8)\
                    .numpy()
            sr_images.append(ndarr)
        return sr_images


if __name__ == "__main__":
    from pprint import pprint
    from matplotlib import pyplot as plt

    image_name = "01DJP58.JPG"

    images = [
        keras_ocr.tools.read(image_name)
    ]
    recognizer = recognize()
    images = recognizer.superresolution(images)
    prediction_groups = recognizer.recognize(images)
    pprint(prediction_groups)
    plate_string = recognizer.get_plate_string(prediction_groups)
    print(plate_string)

    plot_and_save(images, prediction_groups)
