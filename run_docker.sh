#!/bin/bash
export TENSOR_BACKEND="pytorch"  # At the moment, this is only pytorch, we may or may not need to support tensorflow or keras
export DEVICE_BACKEND="cuda"      # Could be cpu or cuda
export IMAGE_BACKEND="skimage"   # Could be skimage or cv2


#Root to the dataset raw material
export IMAGENET_ROOT="/data/Imagenet"
export XRAY_ROOT="/data/xrays"

#Root to the folders necessary for the code
export LOG_ROOT="/data/log"
export TMP_ROOT="/tmp"
export ANNOTATION_ROOT="/data/annotations"

# Make sure the folders exist
mkdir -p "$LOG_ROOT"
mkdir -p "$TMP_ROOT"

cd src
# For profiling use cProfile
# /usr/bin/env python -m cProfile -o /data/log/profile_data.txt  main.py $@
# To analyze the profile install and run
# gprof2dot -f pstats profile_data.txt | dot -Tpng -o prof.png
/usr/bin/env python main.py $@