FROM python:3.8.5
COPY . /face_detection
WORKDIR /face_detection
RUN apt-get update -y
RUN pip install --upgrade pip
RUN pip install --upgrade pillow
RUN apt-get install -y python3-opencv
RUN pip install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.8/index.html
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "app.py"]