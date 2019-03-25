# Custom Object Detection with TensorFlow 

![alt-text](https://github.com/skygate/skyessays/blob/master/tree_detection/output/tree_detection.gif)

## How to detect objects in a pill

### Step 0. Set up your environment 
Be sure that you have installed Python!

- compile the protobuf libraries

`protoc object_detection/protos/*.proto --python_out=.`

- add models and models/slim to your `PYTHONPATH`

`export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim`

### Step 1. Prepare your data in the proper TensorFlow format

`python object_detection/create_tf_record.py`

### Step 2. Download a base model

TensorFlow provides a collection of detection models pre-trained on the e.g. COCO dataset and more. You can find a list [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). 
Extract the files and move all the `model.ckpt` and the `.config` to the your project directory.

### Step 3. Training time

`python object_detection/train.py --logtostderr --train_dir=<any_dir_name> --pipeline_config_path=<your_config_path>`
        
### Step 4. Export the Inference Graph

You can find checkpoints foryour model in `train_dir` directory. Move the `model.ckpt` files with the step number which you want to export to the root of the project and convert them into a frozen graph.

`python object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path <your_config_path> --trained_checkpoint_prefix model.ckpt-STEP_NUMBER --output_directory output_inference_graph`
        
### Step 5. Just test it!

Create a `test_images` directory which contains a set of the testing data and `output/test_images` directory where results will be saved.

`python object_detection/object_detection_runner.py`


#### That's it! Yay!
