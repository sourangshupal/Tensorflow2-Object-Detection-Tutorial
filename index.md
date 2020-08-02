# Tensorflow 2.x  Object Detection ❤⌛


![officialimage](https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/g3doc/img/kites_detections_output.jpg)


**July 10, 2020	TensorFlow 2 meets the Object Detection API (Blog)**

Link to the official Blog :- https://blog.tensorflow.org/2020/07/tensorflow-2-meets-object-detection-api.html

Object Detection Repo :-  https://github.com/tensorflow/models/tree/master/research/object_detection


This release for object detection includes:

1. New binaries for train/eval/export that are eager mode compatible.

2. A suite of TF2 compatible (Keras-based) models; this includes migrations of our most popular TF1 models (e.g., SSD with MobileNet, RetinaNet, Faster R-CNN, Mask R-CNN), as well as a few new architectures for which we will only maintain TF2 implementations: (1) CenterNet - a simple and effective anchor-free architecture based on the recent Objects as Points paper by Zhou et al, and (2) EfficientDet --- a recent family of SOTA models discovered with the help of Neural Architecture Search.

3. COCO pre-trained weights for all of the models provided as TF2 style object-based checkpoints
Access to DistributionStrategies for distributed training: traditionally, we have mainly relied on asynchronous training for our TF1 models. We now support synchronous training as the primary strategy; Our TF2 models are designed to be trainable using sync multi-GPU and TPU platforms

4. Colab demonstrations of eager mode compatible few-shot training and inference

5. First-class support for keypoint estimation, including multi-class estimation, more data augmentation support, better visualizations, and COCO evaluation.



In this post, I am going to the necessary steps for the training of a custom trained model for Tensorflow2 Object Detection.

**It will be a long one but stick till the end for a fruitful result.**

We will be using Google Colab💚. I love to get the tensor computational power of the GPUs.

