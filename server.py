import tensorflow as tf
import cv2
import numpy as np

import posenet
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

import flask

app = flask.Flask(__name__)
sess = None
model_posenet = {
    'model_cfg': None,
    'model_outputs': None
}
model_tfpose  = None

def load_model_posenet():
    global model_posenet, sess
    if sess is not None:
        sess.close()
    sess = tf.Session()
    model_cfg, model_outputs = posenet.load_model(101, sess)
    model_posenet['model_cfg'] = model_cfg
    model_posenet['model_outputs'] = model_outputs

def load_model_tfpose(model='cmu'):
    global model_tfpose, sess

    if sess is not None:
        sess.close()
    
    model_tfpose = TfPoseEstimator(get_graph_path(model), target_size=(432, 368))

def prepare_image_posenet(image, scale_factor=1.0, output_stride=16):
    input_image, display_image, output_scale = posenet.process_input(
        image, scale_factor=scale_factor, output_stride=output_stride)

    return input_image, display_image, output_scale

def postprocess_posenet_ouput(pose_scores, keypoint_scores, keypoint_coords):
    result = {}

    for id, person in enumerate(keypoint_coords):
        dct = {}
        dct["keypoint_coords"] = person.tolist()
        dct["pose_scores"] = pose_scores[id].tolist()
        dct["keypoint_scores"] = keypoint_scores[id].tolist()
        result["person{}".format(id)] = dct
    
    return result

@app.route("/posenet", methods=["POST"])
def predict_posenet():
    output_stride = 16
    scale_factor = 0.7125
    data = {"success": False}
    r = flask.request
    if r.method == "POST":
        if r.data:
            nparr = np.fromstring(r.data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            input_image, display_image, output_scale = prepare_image_posenet(
                img, scale_factor=scale_factor, output_stride=output_stride)
            
            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_posenet['model_outputs'], 
                feed_dict={'image:0': input_image})
            
            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=5,
                min_pose_score=0.15
            )

            keypoint_coords *= output_scale
            data["pose_scores"] = pose_scores.tolist()
            data["keypoint_scores"] = keypoint_scores.tolist()
            data["keypoint_coords"] = keypoint_coords.tolist()
            #data["predictions"] = postprocess_posenet_ouput(pose_scores, keypoint_scores, keypoint_coords)

            data["success"] = True

            #print(data)
    
    return flask.jsonify(data)

@app.route("/tfpose", methods=["POST"])
def predict_tfpose():
    data = {"success": False}
    r = flask.request
    if r.method == "POST":
        if r.data:
            nparr = np.fromstring(r.data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            humans = model_tfpose.inference(img, resize_to_default=False, upsample_size=4.0)

            #print(humans)
            #data['humans'] = humans

            data["success"] = True
    return flask.jsonify(data)


if __name__ == "__main__":
    print("Loading posenet and starting server")
    load_model_posenet()
    #load_model_tfpose()
    app.run(debug=False)

            

