# https://github.com/orestis-z/mask-rcnn-rgbd
# 2D-3D-S
# https://github.com/orestis-z/mask-rcnn-rgbd/tree/master/instance_segmentation/2D-3D-S
import os, sys
import shutil

import numpy as np
import skimage.io
import cv2
from src.objects_dataset import ObjectsDataset

from src.dvs_config import *

NAME = "N-MNIST-DVS-multiple"


def normalize(img):
    img_min = np.min(img)
    img_max = np.max(img)
    if img_min == img_max:
        if img_max == 0:
            return img
        return np.ones(img.shape) * 127.5
    return np.clip((img - img_min) / (img_max - img_min), 0, 1) * 255


class RGBDDatasetMultiple(ObjectsDataset):
    REGENERATE = True  # TODO: SET THIS TO TRUE WHEN YOU WANT TO GENERATE THE DATASET, not needed always

    WIDTH = 64
    HEIGHT = 64

    CLASSES = [(0, 'Zero'),
               (1, 'One'),
               (2, 'Two'),
               (3, 'Three'),
               (4, 'Four'),
               (5, 'Five'),
               (6, 'Six'),
               (7, 'Seven'),
               (8, 'Eight'),
               (9, 'Nine')]

    def __init__(self):
        super().__init__()
        self.frame_paths = []
        self.current_entries = []

    def add_entry(self, frame_path):
        split_path = frame_path.split('\\')
        parent_root = '/'.join(split_path[:3])
        target = split_path[2]
        frame_id = split_path[4][6:]

        depth_path = os.path.join(parent_root, 'depth', 'depth_' + frame_id)
        mask_path = os.path.join(parent_root, 'mask', 'mask_' + frame_id)
        if os.path.isfile(depth_path) and os.path.isfile(mask_path):
            self.current_entries.append((target, frame_id, frame_path, depth_path, mask_path))
            return 1  # All good

        if not os.path.isfile(depth_path) and not os.path.isfile(mask_path):
            print('Warning: No DEPTH and MASK found for ' + frame_path)
        if not os.path.isfile(depth_path):
            print('Warning: No DEPTH found for ' + frame_path)
        elif not os.path.isfile(mask_path):
            print('Warning: No MASK found for ' + frame_path)
        return None  # Something went wrong!

    def generate_frame_paths(self, dataset_dir, skip=1):
        ind = 0
        for number in os.listdir(dataset_dir):
            parent_root = os.path.join(dataset_dir, number)
            frame_dir = os.path.join(parent_root, 'frame')
            for frame in os.listdir(frame_dir):
                if ind % skip == 0:
                    frame_id = frame[6:]
                    frame_path = os.path.join(frame_dir, 'frame_' + frame_id)
                    self.frame_paths.append(frame_path)
                ind += 1

    def generate_new_frame_paths(self, multiple_digits_dataset_folder):
        generated_id, targets = "", []
        for (target, frame_id, _, _, _) in self.current_entries:
            generated_id += "_" + frame_id[:len(frame_id) - 4]
            targets.append(int(target))

        new_frame_path = os.path.join(multiple_digits_dataset_folder, "frame", 'frame' + generated_id + ".png")
        new_depth_path = os.path.join(multiple_digits_dataset_folder, "depth", 'depth' + generated_id + ".png")
        new_mask_path = os.path.join(multiple_digits_dataset_folder, "mask", 'mask' + generated_id + ".png")

        return new_frame_path, new_depth_path, new_mask_path, targets

    def generate_actual_frames(self, new_frame_path, new_depth_path, new_mask_path):
        # Locations for 4 digits of 34x34 in 64x64 images
        offset = 12
        corner_0 = (random.randint(0, offset), random.randint(0, offset))
        corner_1 = (random.randint(0, offset), 32 - random.randint(0, offset))
        corner_2 = (32 - random.randint(0, offset), random.randint(0, offset))
        corner_3 = (32 - random.randint(0, offset), 32 - random.randint(0, offset))
        image_locations = [corner_0, corner_1, corner_2, corner_3]
        # image_locations = [(0, 0), (0, 32), (32, 0), (32, 32)]
        

        new_frame = np.zeros((64, 64, 3), dtype=np.uint8)
        new_depth = np.zeros((64, 64), dtype=np.uint8)
        new_mask = np.zeros((64, 64, 3), dtype=np.uint8)

        for d, (_, _, frame_path, depth_path, mask_path) in enumerate(self.current_entries):
            frame = skimage.io.imread(frame_path)
            # print(frame)
            depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.cvtColor(skimage.io.imread(mask_path), cv2.COLOR_RGBA2BGR)
            h, w, c = frame.shape
            x0, y0, x1, y1 = image_locations[d][1], image_locations[d][0], \
                             image_locations[d][1] + w - 2, image_locations[d][0] + h - 2

            # new_frame[y0:y1, x0:x1] = frame[1:h - 1, 1:w - 1]
            # new_depth[y0:y1, x0:x1] = depth[1:h - 1, 1:w - 1]
            # new_mask[y0:y1, x0:x1] = mask[1:h - 1, 1:w - 1]
            frame_crop = frame[1:h - 1, 1:w - 1]              # (32,32,3)

            # bool-masker: waar de nieuwe digit iets tekent (niet zwart)
            new_pixels = (frame_crop.sum(axis=2) > 0)[..., None]   # shape (32,32,1)

            # overschrijf alleen die pixels; andere blijven ongewijzigd
            new_frame[y0:y1, x0:x1] = np.where(
                new_pixels,          # True  -> neem nieuwe pixel
                frame_crop,          # False -> laat oude waarde staan
                new_frame[y0:y1, x0:x1]
            )
            # new_frame[y0:y1, x0:x1] = np.maximum(new_frame[y0:y1, x0:x1], frame[1:h - 1, 1:w - 1])
            new_depth[y0:y1, x0:x1] = np.maximum(new_depth[y0:y1, x0:x1], depth[1:h - 1, 1:w - 1])
            new_mask[y0:y1, x0:x1] = np.maximum(new_mask[y0:y1, x0:x1], mask[1:h - 1, 1:w - 1])

            # Tweaks the values of the red channel, so we have different colors for each mask
            partial_mask = np.zeros((h - 2, w - 2))
            xs, ys = np.where(mask[1:h - 1, 1:w - 1, 2] > 0)
            partial_mask[xs, ys] = d / 4 * 255
            new_mask[y0:y1, x0:x1, 1] = np.where(
                partial_mask > 0,
                partial_mask,
                new_mask[y0:y1, x0:x1, 1]
            )
            # new_mask[y0:y1, x0:x1, 1] = partial_mask[partial_mask > 0]  # Set the green channel to the partial mask values
            # new_mask[xs, ys, 1] = partial_mask[xs, ys]  
            # new_mask[y0: y1, x0: x1, 1] = partial_mask


        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)

        cv2.imwrite(new_frame_path, new_frame)
        cv2.imwrite(new_depth_path, new_depth)
        cv2.imwrite(new_mask_path, new_mask)
    
    

    def load(self, dataset_dir, subset, skip=1):
        assert (subset == 'training' or subset == 'validation' or subset == 'testing')

        multiple_digits_dataset = dataset_dir + '_multiple'
        if not os.path.exists(multiple_digits_dataset) and self.REGENERATE:
            os.mkdir(multiple_digits_dataset)

        dataset_dir = os.path.join(dataset_dir, subset)
        multiple_digits_dataset_folder = os.path.join(multiple_digits_dataset, subset)
        if os.path.exists(multiple_digits_dataset_folder) and self.REGENERATE:
            shutil.rmtree(multiple_digits_dataset_folder)

        if not os.path.exists(multiple_digits_dataset_folder):
            os.mkdir(multiple_digits_dataset_folder)
            os.mkdir(os.path.join(multiple_digits_dataset_folder, 'frame'))
            os.mkdir(os.path.join(multiple_digits_dataset_folder, 'depth'))
            os.mkdir(os.path.join(multiple_digits_dataset_folder, 'mask'))

        # SET THE SEED HERE SO WE ALWAYS GENERATE THE SAME THING
        random.seed(42)

        # Add classes
        for cls in self.CLASSES:
            self.add_class(NAME, cls[0] + 1, cls[1])

        count = 0
        print(os.listdir(dataset_dir))

        if not self.frame_paths:
            self.generate_frame_paths(dataset_dir)

        ind = 0
        for current_frame_path in self.frame_paths:
            if ind % skip == 0:
                self.current_entries = []
                self.add_entry(current_frame_path)
                for i in range(3):
                    frame_path_other = self.frame_paths[random.randint(0, len(self.frame_paths) - 1)]
                    self.add_entry(frame_path_other)

                new_frame_path, new_depth_path, new_mask_path, targets = \
                    self.generate_new_frame_paths(multiple_digits_dataset_folder)

                if self.REGENERATE:
                    self.generate_actual_frames(new_frame_path, new_depth_path, new_mask_path)

                self.add_image(
                    NAME,
                    image_id=count,
                    path=new_frame_path,
                    depth_path=new_depth_path,
                    mask_path=new_mask_path,
                    width=self.WIDTH,
                    height=self.HEIGHT,
                    targets=targets)
                count += 1

            ind += 1
        print('added {} images for {}'.format(count, subset))

    def load_image(self, image_id, mode="RGBD"):
        """
        Loads each image with it's depth as the Alpha channel
        :param image_id:
        :param mode:
        :return:
        """
        image = super().load_image(image_id)
        if mode == "RGBD":
            depth = skimage.io.imread(self.image_info[image_id]['depth_path'])
            depth = normalize(depth)
            rgbd = np.dstack((image, depth))
            ret = rgbd
        else:
            ret = image

        return ret

    def to_mask(img, instance):
        return (img == instance).astype(np.uint8)

    # vectorize since this was slow in serial execution
    to_mask_v = np.vectorize(to_mask, signature='(n,m),(k)->(n,m)')

    def load_mask(self, image_id):
        """
        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].

        :param image_id:
        :return:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        mask_path = self.image_info[image_id]['mask_path']
        img = cv2.imread(mask_path, -1)

        # https://github.com/alexsax/2D-3D-Semantics/blob/master/assets/utils.py
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]

        img = R * 256 * 256 + G * 256 + B

        instances = np.unique(img.flatten())
        n_instances = len(instances)

        targets = list(np.asarray(self.image_info[image_id]['targets']) + 1)

        masks = np.repeat(np.expand_dims(img, axis=2), n_instances, axis=2)  # bottleneck code
        masks = self.to_mask_v(masks, instances)

        if not n_instances:
            raise ValueError("No instances for image {}".format(mask_path))

        class_ids = np.array([0] + targets)
        return masks, class_ids


if __name__ == '__main__':
    dataset = RGBDDatasetMultiple()
    dataset.load('../data/N_MNIST_images_20ms_skip_50', 'testing')
    dataset.prepare()

    image_ids = np.random.choice(dataset.image_ids, 4)
    for image_id in image_ids:
        image = dataset.load_image(image_id)
        mask, class_ids = dataset.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset.class_names, limit=5)
