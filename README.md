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

Original Repo [tf-pose-estimation] : https://github.com/ildoonet/tf-pose-estimation
