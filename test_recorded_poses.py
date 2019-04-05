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

def draw_keypoints_np(img, keypoint_scores, keypoint_coords, is_norm=False):
    if not is_norm:
        cv_keypoints = []
        for ks, kc in zip(keypoint_scores, keypoint_coords):
            cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))
        out_img = cv2.drawKeypoints(img, cv_keypoints, outImage=np.array([]))
        return out_img
    else:
        #TODO: Implement plotting code for normalized coords
        import matplotlib.pyplot as plt

        for point in keypoint_coords:
            plt.plot(point[1], -point[0], 'bo')
        plt.show()
        return None

def get_norm_coords(keypoint_coords):
    res = []
    for point in keypoint_coords:
        res.append(point[0])
        res.append(point[1])
    return custom_norm(res)

def custom_norm(vector):
    sq = np.square(vector)
    sm = np.sum(sq)
    mag = np.sqrt(sm)
    return np.divide(vector, mag)

def gather(vector):
    return np.reshape(vector, (-1, 2))


def load_pose_data():
    poses_data = {}
    for pose in poses:
        pose_data = {}
        f_path = os.path.join(record_dir, pose)
        #pose_data['pose_scores'] = np.load(os.path.join(f_path, "pose_scores.npy"))[0:1]

        pose_data['keypoint_scores'] = np.load(os.path.join(f_path, "keypoint_scores.npy"))[0, :].tolist()
        
        keypoint_coords = np.load(os.path.join(f_path, "keypoint_coords.npy"))[0, :].tolist()
        keypoint_coords_l2 = gather(get_norm_coords(keypoint_coords)).tolist()
        pose_data['keypoint_coords'] = keypoint_coords
        pose_data['keypoint_coords_l2'] = keypoint_coords_l2

        poses_data[pose] = pose_data
    return poses_data


def do_test():
    data = load_pose_data()
    #print(data)
    for pose in data.keys():
        image = np.zeros((480,640,3), np.uint8)
        image = draw_keypoints_np(
            image, data[pose]['keypoint_scores'][5:], data[pose]['keypoint_coords_l2'][5:], is_norm=True)
        #cv2.imshow(pose, image)
        #cv2.waitKey(0)




if __name__ == "__main__":
    do_test()
