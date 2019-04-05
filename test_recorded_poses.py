import os
import numpy as np
import cv2
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt

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

def draw_keypoints(img, pose_data, is_norm=False):
    keypoint_scores = pose_data['keypoint_scores']
    keypoint_coords = pose_data['keypoint_coords']
    keypoint_coords_l2 = pose_data['keypoint_coords_l2']

    if not is_norm:
        cv_keypoints = []
        for ks, kc in zip(keypoint_scores, keypoint_coords):
            cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))
        out_img = cv2.drawKeypoints(img, cv_keypoints, outImage=np.array([]))
        return out_img
    else:
        #TODO: Implement plotting code for normalized coords

        for point in keypoint_coords_l2:
            plt.plot(point[1], -point[0], 'bo')
        plt.show()
        return None

def get_norm_coords(keypoint_coords):
    return custom_norm(flatten(keypoint_coords))

def custom_norm(vector):
    sq = np.square(vector)
    sm = np.sum(sq)
    mag = np.sqrt(sm)
    return np.divide(vector, mag)

def gather(vector):
    return np.reshape(vector, (-1, 2))

def flatten(vector):
    return np.reshape(vector, (1,-1))

def load_pose_data(trim_head=False):
    poses_data = {}
    for pose in poses:
        pose_data = {}
        f_path = os.path.join(record_dir, pose)

        pose_scores = np.load(os.path.join(f_path, "pose_scores.npy"))
        keypoint_scores = np.load(os.path.join(f_path, "keypoint_scores.npy"))
        keypoint_coords = np.load(os.path.join(f_path, "keypoint_coords.npy"))

        poses_data[pose] = process_pose_data(pose_scores, keypoint_scores, keypoint_coords, trim_head=trim_head)
    return poses_data


def process_pose_data(pose_scores, keypoint_scores, keypoint_coords, trim_head=True):
    pose_data = {}

    pose_scores = pose_scores[0:1].tolist()
    keypoint_scores = keypoint_scores[0, :].tolist()
    keypoint_coords = keypoint_coords[0, :].tolist()
    keypoint_coords_l2 = gather(get_norm_coords(keypoint_coords)).tolist()

    if trim_head:
        pose_data['keypoint_scores'] = keypoint_scores[5:]
        pose_data['keypoint_coords'] = keypoint_coords[5:]
        pose_data['keypoint_coords_l2'] = keypoint_coords_l2[5:]
    else:
        pose_data['keypoint_scores'] = keypoint_scores
        pose_data['keypoint_coords'] = keypoint_coords
        pose_data['keypoint_coords_l2'] = keypoint_coords_l2
    
    return pose_data

def match_pose(pose1, pose2):
    p1_data = flatten(pose1['keypoint_coords_l2'])
    p2_data = flatten(pose2['keypoint_coords_l2'])

    sim = cosine_similarity(p1_data, p2_data)[0]
    dist = np.sqrt(2*(1-sim))

    return dist

def match_pose_weighted(pose1, pose2):
    p1_data = pose1['keypoint_coords_l2']
    p2_data = pose2['keypoint_coords_l2']
    p2_conf = pose2['keypoint_scores']

    sum_p2_conf = np.sum(p2_conf)
    sub = np.subtract(p2_data, p1_data)
    sub_mag = [np.linalg.norm(point) for point in sub]
    conf_sub_mag = np.dot(p2_conf, sub_mag)
    dist = (1.0/sum_p2_conf) * conf_sub_mag

    return dist

def record_and_run(base_pose):
    import requests
    import json

    testing = 'posenet'
    addr = 'http://localhost:5000'
    test_url = addr + '/{}'.format(testing)

    content_type = 'image/jpeg'
    headers = {'content_type': content_type}

    cap = cv2.VideoCapture(1)
    cap.set(3, 640)
    cap.set(4, 480)
    dist = 0.0
    dist2 = 0.0

    base_img = np.zeros((480,640,3), np.uint8)
    base_img = draw_keypoints(
        base_img, base_pose, is_norm=False)
    cv2.imshow('pose', base_img)

    while True:
        _, img = cap.read()
        _, img_encoded = cv2.imencode('.jpg', img)

        response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
        result = json.loads(response.text)

        if not result["success"]:
            continue
        
        pose_scores, keypoint_scores, keypoint_coords = np.array(result['pose_scores']), np.array(result['keypoint_scores']), np.array(result['keypoint_coords'])

        pose_data = process_pose_data(pose_scores, keypoint_scores, keypoint_coords, trim_head=False)

        if (pose_scores[0] > 0.1):
            dist = match_pose(base_pose, pose_data)[0]
            dist2 = match_pose_weighted(base_pose, pose_data)

        ### DRAW CODE
        image = np.zeros((480,640,3), np.uint8)
        image = draw_keypoints(image, pose_data, is_norm=False)

        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        botLeftCorner = (0,50)
        fontScale = 1
        fontColor = (255,255,255)
        lineType = 2

        cv2.putText(image, "c: {}, w: {}".format(str(dist)[:5], str(dist2)[:5]), botLeftCorner, font, fontScale, fontColor, lineType)


        cv2.imshow('video', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            return

def do_test():
    data = load_pose_data(trim_head=False)
    #print(data)
    """
    for pose in data.keys():
        image = np.zeros((480,640,3), np.uint8)
        image = draw_keypoints(
            image, data[pose], is_norm=False)
        cv2.imshow(pose, image)
    """

    record_and_run(data['one-up'])

    cv2.destroyAllWindows()




if __name__ == "__main__":
    do_test()
