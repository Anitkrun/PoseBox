import argparse
import logging
import time

import cv2
import numpy as np
import math

from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

import match_util as mu
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


if __name__ == '__main__':

    """
        Run pose estimation realtime on a webcam.
    """

    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--image', type=str, default='./images/apink1.jpg')
    parser.add_argument('--image2', type=str, default='./images/fem4.jpg')

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    logger.debug('cam read+')
    #cam = cv2.VideoCapture(args.camera)
    #ret_val, image = cam.read()
    #logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    # estimate human poses from a single image !
    image_static = common.read_imgfile(args.image, None, None)
    if image_static is None:
        logger.error('Image can not be read, path=%s' % args.image)
        sys.exit(-1)
    humans = e.inference(image_static, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
    ref_pose = humans[0]
    image_static = TfPoseEstimator.draw_humans(image_static, humans, imgcopy=False)

    image_static2 = common.read_imgfile(args.image2, None, None)
    if image_static2 is None:
        logger.error('Image can not be read, path=%s' % args.image2)
        sys.exit(-1)
    humans = e.inference(image_static2, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
    ref_pose2 = humans[0]
    image_static2 = TfPoseEstimator.draw_humans(image_static2, humans, imgcopy=False)
    

    print(mu.get_cosine_match(ref_pose2, ref_pose))

    mu.plot_points(ref_pose, ref_pose2)

    cv2.imshow('ref pose2', image_static2)
    cv2.imshow('ref pose', image_static)

    cv2.waitKey()
    cv2.destroyAllWindows()

    #while True:
        #ret_val, image = cam.read()

        #logger.debug('image process+')
        #humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        #logger.debug('postprocess+')
        #image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        #if len(humans) > 0:
            #human = humans[0]
            #(_, nox, noy) = mu.get_part_pos(common.CocoPart.Nose, human)
            #(_, nex, ney) = mu.get_part_pos(common.CocoPart.Neck, human)
            #dist = math.sqrt((nox-nex)**2 + (noy-ney)**2)
            #print(mu.get_cosine_match(ref_pose2, ref_pose))

        #logger.debug('{} - ({}, {})'.format(part_pos[0], part_pos[1], part_pos[2]))

        #logger.debug('show+')
        #cv2.putText(image,
        #            "FPS: %f" % (1.0 / (time.time() - fps_time)),
        #            (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #            (0, 255, 0), 2)
        #cv2.imshow('Video feed', image_static2)
        #cv2.imshow('Reference pose', image_static)
        #fps_time = time.time()
        #if cv2.waitKey(1) == 27:
        #    break
        #logger.debug('finished+')

    #cv2.destroyAllWindows()
