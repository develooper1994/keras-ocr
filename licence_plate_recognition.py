import numpy as np
import keras_ocr


def plot_and_save():
    fig, axs = plt.subplots(nrows=len(images), figsize=(20, 20))
    # for ax, image, predictions in zip(axs, images, prediction_groups):
    #     keras_ocr.tools.drawAnnotations(image=image, predictions=predictions, ax=ax)
    keras_ocr.tools.drawAnnotations(image=images[0], predictions=prediction_groups[0], ax=axs)
    plt.show()
    fig.savefig("figure_" + image_name)


def sort_bounindg_boxes(prediction_groups):
    prediction_group = prediction_groups[0]
    arr_prediction_group = np.array(prediction_group)
    bounding_boxes = arr_prediction_group[:, 1]
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
    def __init__(self):
        self.pipeline = keras_ocr.pipeline.Pipeline()

    def __call__(self, images):
        return self.recognize(images)

    def recognize(self, images):
        if not isinstance(images, np.ndarray):
            images = [image for image in images]
        prediction_groups = self.pipeline.recognize(images)
        prediction_groups = sort_bounindg_boxes(prediction_groups)
        return prediction_groups
    
    def get_plate_string(self, predictions, pad_char=''):
        prediction_array = np.array(predictions[0])
        prediction_array_strings = prediction_array[:, 0]
        prediction_strings_list = prediction_array_strings.tolist()
        plate_string = pad_char.join(prediction_strings_list)
        # plate_string = pad_char.join(prediction_strings_list[1:])  # removes 'tr'
        return plate_string


if __name__ == "__main__":
    from pprint import pprint
    from matplotlib import pyplot as plt

    image_name = "01DJP58.JPG"

    images = [
        keras_ocr.tools.read(image_name)
    ]
    recognizer = recognize()
    prediction_groups = recognizer(images)
    pprint(prediction_groups)
    plate_string = recognizer.get_plate_string(prediction_groups)
    print(plate_string)

    plot_and_save()
