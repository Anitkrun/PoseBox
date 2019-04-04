import argparse
import logging
import ntpath
import os
import pickle
import signal
import sys
import time

import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

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
    parser.add_argument('--output', type=str, default='./processed/', help='output dir to write inference data to.')
    parser.add_argument('--mode', type=str, default='play', help='record / play')

    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    args = parser.parse_args()

    return args

def getFileName(path):
    fname = ntpath.basename(path)
    return fname.split('.')[0]

out_data = {}
out_file_path = ''

def pickleData():
    with open(out_file_path, 'wb') as f:
        pickle.dump(out_data, f, pickle.HIGHEST_PROTOCOL)

def record(args):

    # Open output file:
    """
        Output filename: <in_filename>_<model_name>
    """
    global out_data, out_file_path

    in_file = getFileName(args.video)
    out_file = ''.join([in_file, '_', args.model])
    out_file_path = os.path.join(args.output, out_file)

    logger.info(out_file_path)
    #with open(outfile, 'wb') as f:
    frame_idx = 0
    det_frame = 0

    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    cap = cv2.VideoCapture(args.video)

    if cap.isOpened() is False:
        print("Error opening video stream or file.")
    
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    logger.info('Num frames to process: %d' % num_frames)
    pcnt = 0.0

    logger.info('Recording pose data to %s' % out_file_path)
    while cap.isOpened():
        ret, image = cap.read()
        if image is None:
            break
        
        humans = e.inference(image, resize_to_default=True, upsample_size=args.resize_out_ratio)

        if len(humans) > 0:
            out_data[frame_idx] = humans

        pcnt = (frame_idx+1)/num_frames * 100.0
        sys.stdout.write("\rDone {0:.2f}%".format(pcnt))
        sys.stdout.flush()

        frame_idx += 1
    
    cap.release()

    logger.info('Writing data to %s' % out_file)
    pickleData()

def play(args):
    in_data = {}
    in_file = getFileName(args.video)
    data_file = ''.join([in_file, '_', args.model])
    data_file_path = os.path.join(args.output, data_file)

    if not os.path.exists(data_file_path):
        logger.error("No pose data for the video. Run with record flag to record pose data.")
        sys.exit(-1)
    
    with open(data_file_path, 'rb') as f:
        in_data = pickle.load(f)
    
    frame_idx = 0
    
    cap = cv2.VideoCapture(args.video)

    if cap.isOpened() is False:
        print("Error opening video stream or file.")
    while cap.isOpened():
        ret, image = cap.read()
        if image is None:
            logger.error('Image can not be read, path=%s' % args.image)
            sys.exit(-1)
        
        if frame_idx in in_data.keys():
            humans = in_data[frame_idx]
            if not args.showBG:
                image = np.zeros(image.shape)
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        cv2.imshow('tf-pose-estimation-stub', image)

        frame_idx += 1

        if cv2.waitKey(24) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def sigint_handler(sig, frame):
    logger.info('Writing data to %s' % out_file_path)
    pickleData()
    sys.exit(0)

logger = getLogger('TfPoseEstimator-RecUtil')

if __name__ == '__main__':

    """ 
        Utility to write data to disk and load from it.
    """

    args = getArgs()

    if args.mode == 'record':
        signal.signal(signal.SIGINT, sigint_handler)
        record(args)

    if args.mode == 'play':
        play(args)
