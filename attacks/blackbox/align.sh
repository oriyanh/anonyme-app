#!/bin/bash

# Args:
# 1 - MTCNN align venv path
# 2 - MTCNN alignment script path
# 3 - Input images dir
# 4 - Output images dir
# 5 - Image size

# Activate venv
source $1/bin/activate

# Set python path variable to the facenet src directory
# export PYTHONPATH=/cs/ep/503/amit/facenet/src

# remove existing training alignment
rm -rf $4

# align training images
python $2 $3 $4 --image_size $5

