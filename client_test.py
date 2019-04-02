import requests
import json
import cv2
import numpy as np
import time

import posenet

testing = 'posenet'

addr = 'http://localhost:5000'
test_url = addr + '/{}'.format(testing)

content_type = 'image/jpeg'
headers = {'content_type': content_type}

img = cv2.imread('./images/frisbee.jpg')


_, img_encoded = cv2.imencode('.jpg', img)

print("Requesting on {}".format(test_url))

response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)

result = json.loads(response.text)

if testing == 'posenet':
    overlay_image = posenet.draw_skel_and_kp(
        img, np.array(result['pose_scores']), np.array(result['keypoint_scores']), np.array(result['keypoint_coords']),
        min_pose_score=0.15, min_part_score=0.1)

    cv2.imshow('posenet_api', overlay_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_cam():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    start = time.time()
    frame_count = 0
    while True:
        _, img = cap.read()
        _, img_encoded = cv2.imencode('.jpg', img)

        response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
        result = json.loads(response.text)

        if not result["success"]:
            continue
        
        pose_scores, keypoint_scores, keypoint_coords = np.array(result['pose_scores']), np.array(result['keypoint_scores']), np.array(result['keypoint_coords'])

        overlay_image = posenet.draw_skel_and_kp(
                img, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)
        
        cv2.imshow('posenet', overlay_image)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print('Average FPS: ', frame_count / (time.time() - start))

run_cam()