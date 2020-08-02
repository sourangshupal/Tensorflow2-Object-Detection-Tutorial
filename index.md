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

## Let's divide this tutorial into different sections

1. Installation of all necessary libraries
2. Preparing Dataset for Custom Training
3. Using a pretrained model
4. Creating Labelmap.pbtxt
5. Creating xml to csv
6. Creating  tensorflow records files from csv
7. Getting the config file and do the necessary changes
8. Start the training
9. Stop/Resume the training
10. Exporting the graph
11. Doing prediction on the custom trained model
12. Using Webcam for Prediction
13. Working with Videos
14. Applying Multi-threading
15. Creating Docker Images for a Detection App
16. Building a Flask App
17. Building FastAPI App
18. 



### 1. Installation of all necessary libraries
We do need to install the necessary libraries for the execution of the project. Let's open Google Colab first.

Click under File option and then a New Notebook

Wen will follow the reference of the official notebook provided by the community.

Link to the Official Notebook

You can follow the official and Execute all the cells and finally get the results. But I will be creating a notebook and do everything from scratch.



Change the Runtime of the Notebook to GPU

Let's start installing the packages required 

By default, Tensorflow Gpu packages come pre-installed in the environment.

But you can always check by doing 



In the next step follow the execution flow of the official notebook.


    #install tfslim
    pip install tfslim
    #nstall pycocotools
    pip install pycocotools



#### Clone the offical Models Repo 

	import os
	import pathlib
    
    if  "models"  in pathlib.Path.cwd().parts:
       while  "models"  in pathlib.Path.cwd().parts:
        os.chdir('..')
    elif  not pathlib.Path('models').exists():
    	!git clone --depth 1 https://github.com/tensorflow/models


Tensorflow Models  Repository :- [Tensorflow Models Repository](https://github.com/tensorflow/models)



#### Conversion of the protos files to the python files

    cd models/research/
    protoc object_detection/protos/*.proto --python_out=.

#### Installation of the object-detection library

    cd models/research
    pip install .

#### Importing all the necessary packages

    #Import the necessary packages
    import numpy as np
    import os
    import six.moves.urllib as urllib
    import sys
    import tarfile
    import tensorflow as tf
    import zipfile
    from collections import defaultdict
    from io import StringIO
    from matplotlib import pyplot as plt
    from PIL import Image
    from IPython.display import display
    
    #Import the object detection modules
    from object_detection.utils import ops as utils_ops
    from object_detection.utils import label_map_util
    from object_detection.utils import visualization_utils as vis_util



#### Configuring some patched for TF2 from TF1

    #patch tf1 into `utils.ops`
    utils_ops.tf = tf.compat.v1
    #Patch the location of gfile
    tf.gfile = tf.io.gfile

#### Model Downloader and loading fucntion

Model selection can be done from the [Tensorflow 2 Model ZOO](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)

    def  load_model(model_name):
    	base_url =  'http://download.tensorflow.org/models/object_detection/'
    	model_file = model_name +  '.tar.gz'
    	model_dir = tf.keras.utils.get_file(
    	fname=model_name,
    	origin=base_url + model_file,
    	untar=True)
    	model_dir = pathlib.Path(model_dir)/"saved_model"
    	model = tf.saved_model.load(str(model_dir))
    	return model

#### Loading the Labelmap

    #List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS =  'models/research/object_detection/data/mscoco_label_map.pbtxt'
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

#### Going to the detection part

    #If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
    PATH_TO_TEST_IMAGES_DIR = pathlib.Path('models/research/object_detection/test_images')
    TEST_IMAGE_PATHS =  sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
    TEST_IMAGE_PATHS

#### Model Selection from the Model Zoo


[Tensorflow2 Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)


In the model zoo there are various different types of SOTA models available. We can use any one for inference.

	
    model_name =  "efficientdet_d0_coco17_tpu-32"
    detection_model = load_model(model_name)

I am using here [EfficientNet](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz) you can use any one according to your choice.

#### Checking the signature of the models i.e output shapes, datatypes and inputs

    print(detection_model.signatures['serving_default'].inputs)
    detection_model.signatures['serving_default'].output_dtypes
    detection_model.signatures['serving_default'].output_shapes

#### Utilizing the function to load and do prediction

    def run_inference_for_single_image(model, image):
      image = np.asarray(image)
      # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
      input_tensor = tf.convert_to_tensor(image)
      # The model expects a batch of images, so add an axis with `tf.newaxis`.
      input_tensor = input_tensor[tf.newaxis,...]
    
      # Run inference
      model_fn = model.signatures['serving_default']
      output_dict = model_fn(input_tensor)
    
      # All outputs are batches tensors.
      # Convert to numpy arrays, and take index [0] to remove the batch dimension.
      # We're only interested in the first num_detections.
      num_detections = int(output_dict.pop('num_detections'))
      output_dict = {key:value[0, :num_detections].numpy() 
                     for key,value in output_dict.items()}
      output_dict['num_detections'] = num_detections
    
      # detection_classes should be ints.
      output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
       
      # Handle models with masks:
      if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                  output_dict['detection_masks'], output_dict['detection_boxes'],
                   image.shape[0], image.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
        
      return output_dict

#### Running the object detection module some some test images

There is a folder calles test images in the object detection folder with two images.

    def show_inference(model, image_path):
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = np.array(Image.open(image_path))
      # Actual detection.
      output_dict = run_inference_for_single_image(model, image_np)
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          output_dict['detection_boxes'],
          output_dict['detection_classes'],
          output_dict['detection_scores'],
          category_index,
          instance_masks=output_dict.get('detection_masks_reframed', None),
          use_normalized_coordinates=True,
          line_thickness=8)
    
      display(Image.fromarray(image_np))

#### Displaying the final results

    for image_path in TEST_IMAGE_PATHS:
      show_inference(detection_model, image_path)
















### 2. Preparing Dataset for Custom Training

Here we will be using the famous Card Dataset provided by [Edge Electronics](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/tree/master/images) . Here the data is already annotated. So we do not need to do the hard work.

Readers might skip this part as we will talking about the annotation process.

The tool that we will be using is [Labelimg.](https://github.com/tzutalin/labelImg)

Linux Users :- [Follow steps mentioned in the Github Repo](https://github.com/tzutalin/labelImg#ubuntu-linux)

Windows Download Link :- [Download Link](https://www.dropbox.com/s/kqoxr10l3rkstqd/windows_v1.8.0.zip?dl=1)

After the installation is successful. Open the tool 

First Look of the tool

Select Open Directory

Then select the folder containing the images.

Images will be shown in the right below as a list

Let's start annotating

Select the PascalVOC option and not Yolo

Click on Create Rect Box and then annotate the image the object or objects in the image.

Click on Save.

Click on Next and then continue with the same process for each images.

### 3. Using a Pre trained model

From the [Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) we will be selecting the Coco trained [RetinaNet50](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz)

### 4. Creating Labelmap.pbtxt

    item {
      id: 1
      name: 'nine'
    }
    
    item {
      id: 2
      name: 'ten'
    }
    
    item {
      id: 3
      name: 'jack'
    }
    
    item {
      id: 4
      name: 'queen'
    }
    
    item {
      id: 5
      name: 'king'
    }
    
    item {
      id: 6
      name: 'ace'
    }

### 5. Creating xml to csv

    !python xml_to_csv.py

### 6. Creating  tensorflow records files from csv

Changes to be done in the generate_tfrecord.py file

    #TO-DO replace this with label map
    def class_text_to_int(row_label):
        if row_label == 'nine':
            return 1
        elif row_label == 'ten':
            return 2
        elif row_label == 'jack':
            return 3
        elif row_label == 'queen':
            return 4
        elif row_label == 'king':
            return 5
        elif row_label == 'ace':
            return 6
        else:
            None


Execution of the genrate_tfrecord.py file to create tf records.

    #for training data
    python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record
        
    #for validation data
    python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record

### 6. Getting the config file and do the necessary changes

Config file location will be available in the downloaded pretrained folder. 

Pretrained we are using :- [RetinaNet50](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz)

After downloading it. Unzip it and the pipeline.config file will be available.

Open the file in any text editor and do the following changes

 **Change the num_classes to 6. (it is based on the no of classes in the dataset)**

    model {
      ssd {
        num_classes: 6  ## Change Here
        image_resizer {
          fixed_shape_resizer {
            height: 640
            width: 640
          }

*Comments must be removed*

**Change  fine_tune_checkpoint  value to the checkpoint file of the pretrained model, num_steps to your desired number and fine_tune_checkpoint_type value to  "detection " from "classification".**

      fine_tune_checkpoint: "ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0"
      num_steps: 50000    #changed
      startup_delay_steps: 0.0
      replicas_to_aggregate: 8
      max_number_of_boxes: 100
      unpad_groundtruth_tensors: false
      fine_tune_checkpoint_type: "detection"  #changed
      use_bfloat16: true
      fine_tune_checkpoint_version: V2
    }

**Change the path of labelmap.pbtxt ,train.record and test.record**

    train_input_reader {
      label_map_path: "training/labelmap.pbtxt"
      tf_record_input_reader {
        input_path: "train.record"
      }
    }
    eval_config {
      metrics_set: "coco_detection_metrics"
      use_moving_averages: false
    }
    eval_input_reader {
      label_map_path: "training/labelmap.pbtxt"
      shuffle: false
      num_epochs: 1
      tf_record_input_reader {
        input_path: "test.record"
      }
    }

## Google Drive download link for the utility files

**Files provided :-** 
1. xml_to_csv.py
2. genrate_tfrecord.py
3. config file for model
4. Dataset

**Download link :-** 

### 7. Start the training

    !python model_main_tf2.py --model_dir=training --pipeline_config_path=training/pipeline.config

### 8. Stop/Resume the training

    !python model_main_tf2.py --model_dir=training --pipeline_config_path=training/pipeline.config

Stop the training

it can be done by a Keyboard Interrupt or COntrol+C

### 9. Exporting the graph

    python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/pipeline.config
 --trained_checkpoint_prefix training/model.ckpt-25000 --output_directory inference_graph

### 10. Doing prediction on the custom trained model

For Prediction we will be using the notebook at we used for the first time or the one provided in the repository i.e object_detection_tutorial_tf2.ipynb


Thanks to the Wonderful TensorFlow Community












