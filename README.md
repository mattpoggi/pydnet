# PyDnet
This repository contains the source code of pydnet, proposed in the paper "Towards real-time unsupervised monocular depth estimation on CPU", IROS 2018.
If you use this code in your projects, please cite our paper:

```
@inproceedings{pydnet18,
  title     = {Towards real-time unsupervised monocular depth estimation on CPU},
  author    = {Matteo Poggi and
               Filippo Aleotti and
               Fabio Tosi and
               Stefano Mattoccia},
  booktitle = {IEEE/JRS Conference on Intelligent Robots and Systems (IROS)},
  year = {2018}
}
```

## Run pydnet on a webcam

To run pydnet on a webcam, just launch

```
python webcam.py --checkpoint_dir /checkpoint/IROS18/pydnet --resolution [1,2,3]
```

## Train pydnet from scratch

Code for training will be (eventually) uploaded in future.
Meanwhile, you can train pydnet by embedding it into https://github.com/mrharicot/monodepth
