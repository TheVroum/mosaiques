Usage :
1) Images must all be in the same folder, with the names being :
xxxx_yyyy.jpg where xxx is the group number and yyy the image number in the group.
xxxx and yyyy are both integer having any number of digits.
For each group, image numbers must be consecutive.

Only the first 3000 image will be treated (change MAX_FILES to None to remove limit),
and any file not ending with .jpg will be ignored.

2) In produce_descriptors.py, replace /path/to/images_folder/ with the path to images folder (IMAGES_PATH).

3) Run produce_descriptors.py

4) Run produce_indexes.py

5) Run infer.py