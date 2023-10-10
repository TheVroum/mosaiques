from random import sample
from pickle import dumps, loads
from dataclasses import dataclass
from re import match
from os.path import basename
from glob import glob

import numpy as np
from skimage.color import rgb2gray
from skimage.feature import SIFT

import utils



DEFAULT_PICKLE_FILE_PATH = "./descs_2.bin"
MAX_DESCRIPTORS = 1_000
MAX_FILES = 3_000

IMAGES_PATH = "/path/to/images_folder/" # TODO: just change this

@dataclass
class ImageDescriptor:
    keypoints: np.ndarray
    descriptor: np.ndarray
    orientation: np.ndarray
    sigmas: np.ndarray
    def random_truncate(self, max_descriptor: int = MAX_DESCRIPTORS) -> None:
        d_num = self.keypoints.shape[0]
        selector = sample(
            [True]*max_descriptor + [False]*(d_num - max_descriptor),
            k=d_num,)
        self.keypoints = self.keypoints[selector, :]
        self.descriptor = self.descriptor[selector, :]
        self.orientation = self.orientation[selector]
        self.sigmas = self.sigmas[selector]

def path_to_descriptor(path: str, descriptor_extractor = SIFT()) -> ImageDescriptor:
    image = utils.img_to_float(path)
    image = rgb2gray(image)
    descriptor_extractor.detect_and_extract(image)
    return ImageDescriptor(
        descriptor_extractor.keypoints,  # type: ignore
        descriptor_extractor.descriptors,  # type: ignore
        descriptor_extractor.orientations,  # type: ignore
        descriptor_extractor.sigmas,)  # type: ignore

def extract_image_and_descriptors_groups(
    glob_path: str = f"{IMAGES_PATH.removesuffix('/')}/*.jpg",
    max_file: int = MAX_FILES,
    max_descriptor: int = MAX_DESCRIPTORS,
) -> dict[int, list[tuple[int, str, str, ImageDescriptor]]]:
    all_images: dict[int, list[tuple[int, str, str, ImageDescriptor]]] = {}
    for path in glob(glob_path)[:max_file]:
        basename_ = basename(path)
        pm = match(r"([\d]*)_([\d]*)\.jpg", basename_)
        assert pm is not None
        group = int(basename_[slice(*pm.regs[1])])
        index = int(basename_[slice(*pm.regs[2])])
        all_images.setdefault(group, [])
        descriptors = path_to_descriptor(path)
        descriptors.random_truncate(max_descriptor)
        all_images[group].append((index, path, basename_, descriptors,))
    return all_images


def retrieve_image_and_descriptor_from_file(
    path: str = DEFAULT_PICKLE_FILE_PATH,
) -> dict[int, list[tuple[int, str, str, ImageDescriptor]]]:
    with open(path, "rb") as file:
        return loads(file.read())


def produce_and_dump_to(path: str = DEFAULT_PICKLE_FILE_PATH) -> None:
    with open(path, "wb") as file:
        file.write(dumps(extract_image_and_descriptors_groups()))


__all__ = [
    "extract_image_and_descriptors_groups",
    "produce_and_dump_to",
    "ImageDescriptor",
    "retrieve_image_and_descriptor_from_file",
    "MAX_DESCRIPTORS",
]


if __name__ == "__main__":
    produce_and_dump_to(DEFAULT_PICKLE_FILE_PATH)
