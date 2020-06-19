import os
from logging import error
from os import path, walk
from os.path import splitext, basename

import numpy as np
import cv2
from matplotlib import pyplot as plt

from keras_ocr.tools import read
try:
    from . import licence_plate_recognition
except:
    import licence_plate_recognition


class UFPR_parser:
    def __init__(self, data_path, index_maximum=None, crop_plate=True):
        # paths
        self.data_path = data_path
        self.image_paths = []
        self.image_metadata_txt_paths = []
        self.image_metadata_xml_paths = []
        self.image_paths, self.image_metadata_txt_paths, self.image_metadata_xml_paths = self.get_file_paths()

        self.index_maximum = index_maximum
        if index_maximum is None:
            self.index_maximum = len(self.image_paths)

        # metadata
        self.parsed_metadatas = []
        self.parsed_metadatas = self.parse_metadata_txt()

        self.images = []
        self.images = self.read_images(crop_plate=crop_plate)

        self.recognizer = licence_plate_recognition.recognize()
        # prediction_groups = self.recognizer(images)
        # plate_string = self.recognizer.get_plate_string(prediction_groups)

    def __getitem__(self, item):
        assert not item > len(self.images), "!!! index access exceeded !!!"
        image = self.images[item]
        parsed_metadata = self.parsed_metadatas[item]
        return image, parsed_metadata

    def __len__(self):
        return len(self.images)

    def get_file_paths(self):
        for root, dirs, files in walk(self.data_path, topdown=False):
            for name in files:
                filename, file_ext = splitext(basename(name))
                file_ext = file_ext.lower()
                if file_ext == ".png" or file_ext == ".jpg":
                    self.image_paths.append(os.path.join(root, name))
                elif file_ext == ".txt":
                    self.image_metadata_txt_paths.append(path.join(root, name))
                elif file_ext == ".xml":
                    self.image_metadata_xml_paths.append(path.join(root, name))
                else:
                    assert False, "!!! Not supported !!!"
        return self.image_paths, self.image_metadata_txt_paths, self.image_metadata_xml_paths

    def read_images(self, plates_and_positions=None, image_paths=None, crop_plate=True):
        if image_paths is None:
            image_paths = self.image_paths
        if plates_and_positions is None:
            plates_and_positions = self.get_plates_and_positions()
        for image_path, plate_and_position in zip(image_paths[:self.index_maximum], plates_and_positions[:self.index_maximum]):
            image = read(image_path)
            if crop_plate:
                image = self.crop_plate_func(image, plate_and_position)
            self.images.append(image)

        return self.images  # np.stack(images)

    def crop_plate_func(self, image, position_plate):
        # plate = position_plate[0]
        position = position_plate[1]
        bounding_box = self.position_to_bounding_box(position)
        image = image[bounding_box[0, 0]: bounding_box[1, 0], bounding_box[0, 1]: bounding_box[1, 1]]
        return image

    def position_to_bounding_box(self, position):
        coords = [int(coord) for coord in position.split()]
        upper_left = [coords[1], coords[0]]
        lower_right = [coords[1] + coords[3], coords[0] + coords[2]]
        bounding_box = np.array([upper_left, lower_right])
        return bounding_box

    def read_metadata(self, image_metadata_paths=None):
        image_metadatas = []
        if image_metadata_paths is None:
            image_metadata_paths = self.image_metadata_txt_paths
        for image_metadata_path in image_metadata_paths:
            with open(image_metadata_path, "r") as image_metadata_file:
                image_metadata = image_metadata_file.readlines()
            image_metadatas.append(image_metadata)
        return image_metadatas

    def parse_metadata_txt(self, image_metadatas=None):
        if image_metadatas is None:
            image_metadatas = self.read_metadata()
        all_info = {}
        all_info_list = []
        for image_metadata in image_metadatas[:self.index_maximum]:
            # headlines
            camera = image_metadata[0].split(':')[-1][1:-1]
            position_vehicle = image_metadata[1].split(':')[-1][1:-2]
            # vehicle
            type = image_metadata[2].split(':')[-1][1:-1]
            make = image_metadata[3].split(':')[-1][1:-1]
            model = image_metadata[4].split(':')[-1][1:-1]
            year = image_metadata[5].split(':')[-1][1:-1]
            # plate
            plate = image_metadata[6].split(':')[-1][1:-1]
            position_plate = image_metadata[7].split(':')[-1][1:-1]
            # characters
            char_bbs = [meta.split(':')[-1][1:-1] for meta in image_metadata[8:]]

            all_info = {
                "camera": camera,
                "position_vehicle": position_vehicle,
                "type": type,
                "make": make,
                "model": model,
                "year": year,
                "plate": plate,
                "position_plate": position_plate,
                "char_bbs": char_bbs,
            }
            all_info_list.append(all_info)
        self.parsed_metadatas = all_info_list
        return all_info_list

    def get_plates_and_positions(self):
        plates_and_position = []
        for image_metadata in self.parsed_metadatas:
            # plate
            plate = image_metadata["plate"]
            position_plate = image_metadata["position_plate"]
            plates_and_position.append([plate, position_plate])
        return plates_and_position

    def get_predictions(self, images=None):
        if images is None:
            images = self.images
        prediction_groups = self.recognizer(images)
        plate_strings = self.recognizer.get_plate_string(prediction_groups)

        return plate_strings


if __name__ == "__main__":
    data_path = "../UFPR-ALPR dataset/"
    parser = UFPR_parser(data_path=data_path, index_maximum=5)
    print(parser[0])
    plate_strings = parser.get_predictions()
