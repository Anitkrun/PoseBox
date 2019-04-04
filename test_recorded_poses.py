import os
import numpy as np
import cv2
from sklearn import preprocessing

#import posenet

record_dir = './records'
poses = [
    'one-side',
    'one-up',
    'st-pose',
    'tri-pose'
]

"""
DATA FORMAT:
    {
        "one_side": {
            "keypoint_scores": [s1, s2, s3,..., s12],
            "keypoint_coords": [
                [x1,y1],
                [x2,y2],
                [x3,y3],
                .
                .
                [x12,y12]
            ]
        },
        "one-up": {
            ....
        },
        ....
    }

"""

def draw_keypoints_np(img, keypoint_scores, keypoint_coords):
    cv_keypoints = []
    for ks, kc in zip(keypoint_scores, keypoint_coords):
        cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))
    out_img = cv2.drawKeypoints(img, cv_keypoints, outImage=np.array([]))
    return out_img


def load_pose_data():
    poses_data = {}
    for pose in poses:
        pose_data = {}
        f_path = os.path.join(record_dir, pose)
        #pose_data['pose_scores'] = np.load(os.path.join(f_path, "pose_scores.npy"))[0:1]

        pose_data['keypoint_scores'] = np.load(os.path.join(f_path, "keypoint_scores.npy"))[0, :].tolist()

        #pose_data['keypoint_coords'] = np.load(os.path.join(f_path, "keypoint_coords.npy"))[0, :, :].tolist()
        coords = np.load(os.path.join(f_path, "keypoint_coords.npy"))[0, :].tolist()
        coords_l2 = preprocessing.normalize(coords, norm='l2').tolist()
        pose_data['keypoint_coords'] = coords_l2

        poses_data[pose] = pose_data
    return poses_data


def do_test():
    data = load_pose_data()

    for pose in data.keys():
        image = np.zeros((480,640,3), np.uint8)
        image = draw_keypoints_np(
            image, data[pose]['keypoint_scores'][5:], data[pose]['keypoint_coords'][5:])
        cv2.imshow(pose, image)
        cv2.waitKey(0)




if __name__ == "__main__":
    do_test()
