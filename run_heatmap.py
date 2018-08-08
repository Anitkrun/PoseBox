import argparse
import logging
import os
import pickle
import sys
import time

import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

MODE_RECORD = 0
MODE_REPLAY = 1

def getLogger(name, level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

def getArgs():
    parser = argparse.ArgumentParser(description='tf-pose-estimation stub utility')
    parser.add_argument('--video', type=str, default='./videos/video1.mp4')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')

    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    args = parser.parse_args()

    return args

logger = getLogger('TfPoseEstimator-Stub')

if __name__ == '__main__':

    """ 
        Stub utility to write data to disk and load from it.
    """

    args = getArgs()
    cap = cv2.VideoCapture(args.video)

    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    
    while cap.isOpened():
        ret, image = cap.read()
        if image is None:
            logger.error('Image can not be read, path=%s' % args.image)
            sys.exit(-1)

        #t = time.time()
        humans = e.inference(image, resize_to_default=True, upsample_size=args.resize_out_ratio)
        #elapsed = time.time() - t

        heatmp = e.pafMat.transpose((2, 0, 1))
        hm_odd = np.amax(np.absolute(heatmp[::2, :, :]), axis=0)
        hm_even = np.amax(np.absolute(heatmp[1::2, :, :]), axis=0)

        if not args.showBG:
            image = np.zeros(image.shape)
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        cv2.imshow('tf-pose-estimation-stub', image)
        cv2.imshow('tf-pose-estimation-heatmap-x', hm_odd)
        cv2.imshow('tf-pose-estimation-heatmap-y', hm_even)

        if cv2.waitKey(24) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
