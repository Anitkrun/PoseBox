# PoseBox

Toolbox that contains utilities to implement two dimensional pose estimation to analyse and improve various body movement based tasks.

## Dependencies

You need dependencies below.

- python3
- tensorflow 1.4.1+
- opencv3, protobuf, python3-tk

### Opensources

- slim
- slidingwindow
  - https://github.com/adamrehn/slidingwindow
 

### Install

Clone the repo and install 3rd-party libraries.

```bash
$ git clone https://www.github.com/arnitkun/PoseBox
$ cd tf-openpose
$ pip3 install -r requirements.txt
```

Build c++ library for post processing. See : https://github.com/arnitkun/PoseBox/tree/master/tf_pose/pafprocess
```
$ cd tf_pose/pafprocess
$ swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace
```

### Package Install

Alternatively, you can install this repo as a shared package using pip.

```bash
$ git clone https://www.github.com/arnitkun/PoseBox
$ cd PoseBox
$ python setup.py install
```

### How to use record utility

- Place the video file into the video directory
- Run the following command
  ```bash
  $ python run_record_util.py --model mobilenet_thin --mode record --video ./video/<video_file_name>
  ```
  - --model flag specifies which model to use cmu | mobilenet_thin
  - --mode  flag specifies record or play mode
  - --video flag specifies the path to the video
  - --output flag specifies the path to the output record file 
- The output data will be written to ./processed/<video_file_name>_<model_name>
  - TIP: Pressing Ctrl+C will cause the program to dump the recorded data to file and exit.
- Run the above command with the --mode play flag to play the recorded data from file.




**Stuff to be added**

Original Repo [tf-pose-estimation] : https://github.com/ildoonet/tf-pose-estimation
