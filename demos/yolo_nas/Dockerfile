# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags
FROM nvcr.io/nvidia/pytorch:22.08-py3

RUN apt-get update && \
    apt-get install -y libgl1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install -r https://raw.githubusercontent.com/Deci-AI/super-gradients/master/requirements.txt && \
    # For some reason it doesn't work with latest version of OpenCV \ --> AttributeError: partially initialized module 'cv2' has no attribute '_registerMatType' (most likely due to a circular import)
    pip install opencv-python==4.5.5.64

RUN pip install git+https://github.com/tryolabs/norfair.git@master#egg=norfair

RUN pip install super-gradients


WORKDIR /demo/src/