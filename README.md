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
3. Connecting Google Drive with Colab
4. Using a pretrained model
5. Creating Labelmap.pbtxt
6. Creating xml to csv
7. Creating  tensorflow records files from csv
8. Getting the config file and do the necessary changes
9. Start the training
10. Stop/Resume the training
11. Evalauating the model
12. Exporting the graph
13. Doing prediction on the custom trained model
14. Using Webcam for Prediction
15. Working with Videos
16. Converting to Tflite
17. Creating Docker Images for a Detection App **[TODO]**
18. Building a Flask App **[TODO]**
19. Building FastAPI App **[TODO]**
20. Applying Multithreading 



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

    pip freeze
<img src="https://i.ibb.co/7yS41PS/Screenshot-from-2020-08-03-20-31-36.png" alt="Screenshot-from-2020-08-03-20-31-36" border="0">

In the next step follow the execution flow of the official notebook.


    #install tfslim
    pip install tfslim
    #nstall pycocotools
    pip install pycocotools


<img src="https://i.ibb.co/G9sPLsN/Screenshot-from-2020-08-03-20-28-38.png" alt="Screenshot-from-2020-08-03-20-28-38" border="0">

#### Clone the offical Models Repo 

	import os
	import pathlib
    
    if  "models"  in pathlib.Path.cwd().parts:
       while  "models"  in pathlib.Path.cwd().parts:
        os.chdir('..')
    elif  not pathlib.Path('models').exists():
    	!git clone --depth 1 https://github.com/tensorflow/models


<img src="https://i.ibb.co/xS9zR4n/Screenshot-from-2020-08-03-20-33-02.png" alt="Screenshot-from-2020-08-03-20-33-02" border="0">

Tensorflow Models  Repository :- [Tensorflow Models Repository](https://github.com/tensorflow/models)

<img src="https://i.ibb.co/XJycdsf/Screenshot-from-2020-08-03-20-34-29.png" alt="Screenshot-from-2020-08-03-20-34-29" border="0">

#### Conversion of the protos files to the python files

    cd models/research/
    protoc object_detection/protos/*.proto --python_out=.

<img src="https://i.ibb.co/HVbpcmh/Screenshot-from-2020-08-03-20-37-13.png" alt="Screenshot-from-2020-08-03-20-37-13" border="0">

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

<img src="https://i.ibb.co/sFdFf4B/Screenshot-from-2020-08-03-20-43-49.png" alt="Screenshot-from-2020-08-03-20-43-49" border="0">

#### Model Downloader and loading fucntion

Model selection can be done from the [Tensorflow 2 Model ZOO](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)

    def  load_model(model_name):
    	base_url =  'http://download.tensorflow.org/models/object_detection/tf2/20200711/'
    	model_file = model_name +  '.tar.gz'
    	model_dir = tf.keras.utils.get_file(
    	fname=model_name,
    	origin=base_url + model_file,
    	untar=True)
    	model_dir = pathlib.Path(model_dir)/"saved_model"
    	model = tf.saved_model.load(str(model_dir))
    	return model

<img src="https://i.ibb.co/DkxVDdP/Screenshot-from-2020-08-03-20-45-28.png" alt="Screenshot-from-2020-08-03-20-45-28" border="0">

#### Loading the Labelmap

    #List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS =  'models/research/object_detection/data/mscoco_label_map.pbtxt'
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)



#### Going to the detection part

    #If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
    PATH_TO_TEST_IMAGES_DIR = pathlib.Path('models/research/object_detection/test_images')
    TEST_IMAGE_PATHS =  sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
    TEST_IMAGE_PATHS


<img src="https://i.ibb.co/kJV5V4p/Screenshot-from-2020-08-03-20-46-27.png" alt="Screenshot-from-2020-08-03-20-46-27" border="0">

#### Model Selection from the Model Zoo


[Tensorflow2 Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)

<img src="https://i.ibb.co/41yn3Sj/Screenshot-from-2020-08-03-20-48-32.png" alt="Screenshot-from-2020-08-03-20-48-32" border="0">

In the model zoo there are various different types of SOTA models available. We can use any one for inference.

	
    model_name =  "efficientdet_d0_coco17_tpu-32"
    detection_model = load_model(model_name)

<img src="https://i.ibb.co/Zg9GPLP/Screenshot-from-2020-08-03-20-51-11.png" alt="Screenshot-from-2020-08-03-20-51-11" border="0">

I am using here [EfficientNet](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz) you can use any one according to your choice.

#### Checking the signature of the models i.e output shapes, datatypes and inputs

    print(detection_model.signatures['serving_default'].inputs)
    detection_model.signatures['serving_default'].output_dtypes
    detection_model.signatures['serving_default'].output_shapes

<img src="https://i.ibb.co/RcqYjyP/Screenshot-from-2020-08-03-20-52-14.png" alt="Screenshot-from-2020-08-03-20-52-14" border="0">

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


<img src="https://i.ibb.co/Dgwb5xN/Screenshot-from-2020-08-03-20-53-32.png" alt="Screenshot-from-2020-08-03-20-53-32" border="0">


#### Running the object detection module some some test images



There is a folder called **test images** in the object detection folder with two images.

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

<img src="https://i.ibb.co/t3ZFgtP/Screenshot-from-2020-08-03-20-54-54.png" alt="Screenshot-from-2020-08-03-20-54-54" border="0">

#### Displaying the final results

    for image_path in TEST_IMAGE_PATHS:
      show_inference(detection_model, image_path)

<img src="https://i.ibb.co/FVdmvQh/Screenshot-from-2020-08-03-20-56-10.png" alt="Screenshot-from-2020-08-03-20-56-10" border="0">

<img src="https://i.ibb.co/GPynRC9/Screenshot-from-2020-08-03-20-57-24.png" alt="Screenshot-from-2020-08-03-20-57-24" border="0">














### 2. Preparing Dataset for Custom Training

Here we will be using the famous Card Dataset provided by [Edge Electronics](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/tree/master/images) . Here the data is already annotated. So we do not need to do the hard work.

***Readers might skip this part as we will talking about the annotation process.***

The tool that we will be using is [Labelimg.](https://github.com/tzutalin/labelImg)

Linux Users :- [Follow steps mentioned in the Github Repo](https://github.com/tzutalin/labelImg#ubuntu-linux)

Windows Download Link :- [Download Link](https://www.dropbox.com/s/kqoxr10l3rkstqd/windows_v1.8.0.zip?dl=1)

After the installation is successful. Open the tool 

First Look of the tool
<img src="https://i.ibb.co/xmtyYpr/Screenshot-from-2020-08-03-19-31-06.png" alt="Screenshot-from-2020-08-03-19-31-06" border="0">


Select Open Directory and then select the folder containing the images.

Images will be shown in the right below as a list

Let's start annotating

<a href="https://ibb.co/vdHzhMj"><img src="https://i.ibb.co/FYK7XdD/Screenshot-from-2020-08-03-19-32-09.png" alt="Screenshot-from-2020-08-03-19-32-09" border="0"></a>

Select the PascalVOC option and not Yolo

Click on Create Rect Box and then annotate the image the object or objects in the image.

Click on Save.

Click on Next and then continue with the same process for each images.

### 3. Connecting Google Drive with Colab

Here we will be connecting the Google Drive with Google Colab. So that our training checkpoints can be saved in the drive in the runtine disconnection happens because we know it has a limit of around 8-12 hours.

    from google.colab import drive
    drive.mount('/content/drive')

Then click on the provided url and paste the key provided. Your Google Drive will be mounted. in the content folder the drive will be mounted.

<img src="https://i.ibb.co/Dt5Lcjw/Screenshot-from-2020-08-03-21-31-58.png" alt="Screenshot-from-2020-08-03-21-31-58" border="0">

**I will be creating a new folder in Google Drive called TFOD2. Then i will clone the models repository in the TFOD2 for training and future refernce of the model checkpoints.** 
**I will be keeping my complete repository and the folder structure in the the TFOD2 folder.**

A sample picture is provided below :-

<img src="https://i.ibb.co/8cRXJ1J/Screenshot-from-2020-08-03-21-41-37.png" alt="Screenshot-from-2020-08-03-21-41-37" border="0">

<img src="https://i.ibb.co/Jd8gfVt/Screenshot-from-2020-08-03-21-45-17.png" alt="Screenshot-from-2020-08-03-21-45-17" border="0">

<img src="https://i.ibb.co/S37VSQ0/Screenshot-from-2020-08-03-21-46-52.png" alt="Screenshot-from-2020-08-03-21-46-52" border="0">

<img src="https://i.ibb.co/NLv8QvP/Screenshot-from-2020-08-03-21-47-36.png" alt="Screenshot-from-2020-08-03-21-47-36" border="0">



### 4. Using a Pre trained model

From the [Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) we will be selecting the Coco trained [RetinaNet50](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz)

<img src="https://i.ibb.co/tp9CpL6/Screenshot-from-2020-08-03-19-46-23.png" alt="Screenshot-from-2020-08-03-19-46-23" border="0">

### 5. Creating Labelmap.pbtxt

Create a file called **labelmap.pbtxt** where we will be keeping the name of the classes in our Cards Dataset.

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

The file **labelmap.pbtxt** is available in the **utility_files.zip** provided by the Google drive link.


### 6. Creating xml to csv


    !python xml_to_csv.py

The file **xml_to_csv.py** is available in the **utility_files.zip** provided by the Google drive link.


### 7. Creating  tensorflow records files from csv

The file **generate_tfrecord.py** is available in the **utility_files.zip** provided by the Google drive link.


Changes to be done in the generate_tfrecord.py file as per the classes in your dataset.

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

**If you get a None TypeError** in the elif ladder change the value of else  from None to **return 0**

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
            return 0


### 8. Getting the config file and do the necessary changes

Config file location will be available in the downloaded pretrained folder. 

Pretrained we are using :- [RetinaNet50](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz)

After downloading it. Unzip it and the **pipeline.config** file will be available.

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

**Download link :-**  [utility_files.zip](https://drive.google.com/file/d/1PBOgn5rlMx9tjjvfqlAJM8KY-8w05yXg/view?usp=sharing)


### *We will be creating a new folder called as training inside the object_detection folder where we will be keeping two files pipeline.config and labelmap.pbtxt*

<img src="https://i.ibb.co/1rR903n/Screenshot-from-2020-08-04-18-43-16.png" alt="Screenshot-from-2020-08-04-18-43-16" border="0">

My **training** folder looks above in the object detection.

### 9. Start the training

So we are ready to start the training.

**model_main_tf2.py** is the file needed to start the training.

    !python model_main_tf2.py --model_dir=training --pipeline_config_path=training/pipeline.config

<img src="https://i.ibb.co/0FkH3VT/Screenshot-from-2020-08-04-13-08-00.png" alt="Screenshot-from-2020-08-04-13-08-00" border="0">

I have trained for **50000 steps**.

We will be saving all the checkpoints in the **training** folder.

### 10. Stop/Resume the training

    !python model_main_tf2.py --model_dir=training --pipeline_config_path=training/pipeline.config

**Stop the training**

It can be stopped by a **Keyboard Interrupt** or **Control+C**

### 11.Evaluating the Model

    !python model_main_tf2.py --model_dir=training --pipeline_config_path=training/pipeline.config --checkpoint_dir=training

<img src="https://i.ibb.co/kHkt0sX/Screenshot-from-2020-08-04-13-05-13.png" alt="Screenshot-from-2020-08-04-13-05-13" border="0">

### 12. Exporting the graph

    !!python exporter_main_v2.py --input_type image_tensor --pipeline_config_path training/pipeline.config --trained_checkpoint_dir training/ --output_directory exported-models/my_model

<img src="https://i.ibb.co/Htnqqdn/Screenshot-from-2020-08-04-18-50-29.png" alt="Screenshot-from-2020-08-04-18-50-29" border="0">

At the end you can see something similar

<img src="https://i.ibb.co/vYDLmBK/Screenshot-from-2020-08-04-18-50-50.png" alt="Screenshot-from-2020-08-04-18-50-50" border="0">


### 13. Doing prediction on the custom trained model

For Prediction we will be using the notebook at we used for the first time or the one provided in the repository i.e object_detection_tutorial_tf2.ipynb

<img src="https://i.ibb.co/dbdZRp6/Screenshot-from-2020-08-04-18-54-37.png" alt="Screenshot-from-2020-08-04-18-54-37" border="0">

<img src="https://i.ibb.co/FnvRRtL/Screenshot-from-2020-08-04-18-55-18.png" alt="Screenshot-from-2020-08-04-18-55-18" border="0">

<img src="https://i.ibb.co/98G8PhF/Screenshot-from-2020-08-04-18-56-58.png" alt="Screenshot-from-2020-08-04-18-56-58" border="0">

**Final Results**

<img src="https://i.ibb.co/4mxY5Rt/Screenshot-from-2020-08-04-18-58-08.png" alt="Screenshot-from-2020-08-04-18-58-08" border="0">


### 14. Working with Videos
### 15. Converting to Tflite
### 16. Creating Docker Images for a Detection App **[TODO]**
### 17. Building a Flask App **[TODO]**
### 18. Building FastAPI App **[TODO]**
### 19. Applying Multithreading **[TODO]**


***Thanks to the Wonderful TensorFlow Community***
