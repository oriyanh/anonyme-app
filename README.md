
♪ README README README a man of the midnight! ♪

# anonyME

<br>
Amit Segal & Oriyan Hermoni<br>
<br>
**anonyME** is a system which allows users to inoculate their personal images against unauthorized machine learning models, with minimal distortion of the input image, without wearing specialized clothing articles, and with no requirement of any previous knowledge in machine learning or face recognition.

## Requirements & Setup

Project requires Python 3.7, and is implemented using Tensorflow 1.14.0 and Keras 2.3.x.<br>
<br>
In order to install project dependencies, using a new virtual environment, run:

    pip -r install requirements.txt
    pip -r install no_deps_requirements.txt --no-deps

## Google Drive Folder

All weights and graphs can be found in our Google Drive folder found [here](https://drive.google.com/drive/u/1/folders/1PIufZmHXbAw28SleVHKualghO5t0fcrU). This is a private Google Drive folder, access will be given per request.

## Entry Points & Usage

### Web Service

Both our white-box attack pipeline and black-box attack pipelines can be accessed through our web service (currently Proof of Concept).
It can be run by running app.py in the repo's root directory, this requires our whitebox and blackbox weights present under the correct name in the correct directories, relative to the repo's root directory (namely, its parent directory).

### White-box attack pipeline

Our white-box attack pipeline is fully integrated into our web-service, and can only be run from there.

For the code to run correctly, facenet_model.pb must be downloaded from our Drive folder (located in weights directory), and placed in the repo's parent directory.

Running instructions:

A request must be sent to /whitebox endpoint. Its body contains an input file and a target file. It can also contain a json containing eps and num_iter params, where eps is the FGSM attack step size, and num_iter is the maximum number of iteration for which FGSM attack will run.

In order to view the FGSM attack code, you can view attacks.whitebox.fgsm.adversary.py.

### Substitute model training

To train our substitute model, you must:

 1. Create 'dataset/augmented_images' and 'weights' directories in the repo's parent directory.
 2. Pre-process your training set by running `python <repo>/scripts/sort_augmented_images.py <pruning_threshold> <training_directory>`
 3. Then, for every iteration of training, run first `python <repo>/attacks/blackbox/train.py`, then `python <repo>/attacks/blackbox/augmentation.py`, and finally `  
python <repo>/scripts/sort_augmented_images.py <pruning_threshold>`
 4. Your trained model will appear in weights path.

NOTES: 

 - First iteration of training trains a model on the initial dataset of VGGFace2 found in \<training_directory\>, therefore, in order to train a model on a data set that's been augmented 3 times, step 3 must be performed 4 times.
 - An example script which trains a substitute model with 3 iterations of augmentation can be found in scripts.train.sh 

### Evaluate black-box attack

**Requirements**: create 'dataset/attack_eval_res' directory in the repo's parent directory.

In order to evaluate the black-box attack using a trained RESNET50 substitute model's, run `python <repo>/scripts/eval_blackbox_attack.py resnet50 <substitute_weights_path> <substitute_classes_num> <label> <eval_dataset_directory>`

Where:

 - \<substitute_weights_path\> is the directory containing your RESNET50 substitute model's weights in .h5 format
 - \<substitute_classes_num\> is the number of classes in your substitute model
 -  \<label\> is a label for your run. This will be used in the directory name containing the evaluation results, and the graph file names.
 - \<eval_dataset_directory\> is your validation data set directory, formatted as VGGFace2 data set is formatted.

Optionally, you may add:

 - `--blackbox-architecture` in order to change blackbox architecture, possible values are 'RESNET50' or 'SENET50'. This can be used to evaluate the black-box attack using SENET50 as your black-box model, instead of the default RESNET50.
 - `--batch-size` accepts a positive integer, default is 4
 - `--step-size` accepts a positive float, default is 0.004
 - `--max-iter` accepts a positive integer, default is 50

NOTE: an example script for evaluating a blackbox substitute can be found in scripts.attack_eval_resnet50.sh - it uses the substitute model found in Drive folder under weights/resnet50_weights_step5/substitute_resnet50_4.h5, which spans 86 classes.

In order to evaluate the black-box attack as a baseline, run `python <repo>/scripts/eval_blackbox_attack_baseline.py resnet50 <label>  <eval_dataset_directory>`

Where:

-  \<label\> is a label for your run. This will be used in the directory name containing the evaluation results, and the graph file names
 - \<eval_dataset_directory\> is your validation data set directory, formatted as VGGFace2 data set is formatted.

Optionally, you may add:

 - `--batch-size` accepts a positive integer, default is 4
 - `--step-size` accepts a positive float, default is 0.004
 - `--max-iter` accepts a positive integer, default is 50
 - `--cross` is a flag which uses SENET50 as the black-box model. This can be used to evaluate black-box attack using fully trained RESNET50 as the substitute and fully trained SENET50 as the black-box model.
