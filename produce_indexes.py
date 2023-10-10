from math import log10
from os import remove
from sqlite3 import connect
from pickle import dumps

import numpy as np

from faiss import index_factory

from produce_descriptors import (
    retrieve_image_and_descriptor_from_file,
    MAX_DESCRIPTORS,
    ImageDescriptor,)


INDEX_PICKLE_FILE_PATH = "./index_1.bin"
REVERSE_INDEX_DESCRIPTOR_FILE = "./rev_index_1.db"


Q_CREATION = """
CREATE TABLE rev(
    cum INTEGER PRIMARY KEY,
    group_id INTEGER,
    image_local_id INTEGER,
    UNIQUE (group_id, image_local_id)
);
INSERT INTO rev VALUES (0, NULL, NULL);
"""

Q_ADD_IMAGE = """
INSERT INTO rev VALUES (? + (SELECT cum FROM rev ORDER BY cum DESC LIMIT 1), ?, ?);
"""

Q_GET_IMAGE = """
SELECT group_id AS group_, image_local_id AS number FROM rev WHERE cum > ? ORDER BY cum ASC LIMIT 1;
"""



MAX_EXPECTED_POINTS_OF_INTEREST = 1000
IMAGES_PER_POINT_OF_INTEREST = 3


OFFSET = -2
CLUSTER_NUMBER_POW = 2*log10(
    MAX_EXPECTED_POINTS_OF_INTEREST*
    IMAGES_PER_POINT_OF_INTEREST*
    MAX_DESCRIPTORS) + OFFSET


def main():

    # index chosen according to https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
    # Scaling to bigger zones : just update MAX_EXPECTED_POINTS_OF_INTEREST to adapt the index.
    index = index_factory(128, f"PCA64,IVF{2**round(CLUSTER_NUMBER_POW)}_HNSW32,Flat")

    image_and_descriptors = retrieve_image_and_descriptor_from_file()

    descs = np.concatenate([
        i_d[3].descriptor
        for gr in image_and_descriptors.values()
        for i_d in gr])

    try:
        with open(REVERSE_INDEX_DESCRIPTOR_FILE, "x"):
            pass
    except FileExistsError:
        remove(REVERSE_INDEX_DESCRIPTOR_FILE)

    with connect(REVERSE_INDEX_DESCRIPTOR_FILE) as conn:
        c = conn.cursor()
        c.executescript(Q_CREATION)
        for group, (number, _, _, image_descriptor) in [
            (key, element)
            for key, value in image_and_descriptors.items()
            for element in value
        ]:
            c.execute(Q_ADD_IMAGE, (image_descriptor.descriptor.shape[0], group, number))

    index.train(descs)

    index.add(descs)

    with open(INDEX_PICKLE_FILE_PATH, "wb") as output_file:
        output_file.write(dumps(index))

if __name__ == "__main__":
    main()

__all__ = [
    "Q_GET_IMAGE",
]
