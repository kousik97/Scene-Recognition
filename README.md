# Scene-Recognition
Demo of VGG-16 scene recognition model pre-trained on places-365 dataset.

Note : This is just a demo script to visualize quick results of scene contexts on images.
The pre-trained models are all available in the MIT Places website.

## Input-Output 
### Input 
Image 
### Output 
Top 5 scene contexts with the associated probability

## Requirements
* python2
* Caffe
* Pandas

All codes are tested on a container built from Ubuntu 14.04 CPU/GPU docker image downloaded from floyd-hub(link given below).

## Demo
1. Download the Scene_Recognition_models directory from [Drive](https://drive.google.com/open?id=0ByDWS1KXv3socHZvd2ZJT05kZEk) and place it in the same level as that of Scene.py
2. To find scene context of an image, run python Scene.py -i /full/path/to/image
3. To find scene contexts of all images inside a directory, run python Scene.py -d /path/to/directory. This will store the results in result.csv file

For quick results, use the sample images provided in the example_images directory.


## References
* The model/weight files are downloaded from [Scene-365 Model zoo](https://github.com/CSAILVision/places365)
* [Docker Image](https://github.com/floydhub/dl-docker)
* [MIT Places Dataset](http://places.csail.mit.edu/)
