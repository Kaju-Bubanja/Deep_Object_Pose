import yaml
import numpy as np


def load_params():
    config_name = "config_pose.yaml"
    yaml_path = 'config/{}'.format(config_name)
    with open(yaml_path, 'r') as stream:
        try:
            print("Loading DOPE parameters from '{}'...".format(yaml_path))
            params = yaml.load(stream)
            print('    Parameters loaded.')
            return params
        except yaml.YAMLError as exc:
            print(exc)


def get_config_options(params):
    matrix_camera = np.zeros((3, 3))
    matrix_camera[0, 0] = params["camera_settings"]['fx']
    matrix_camera[1, 1] = params["camera_settings"]['fy']
    matrix_camera[0, 2] = params["camera_settings"]['cx']
    matrix_camera[1, 2] = params["camera_settings"]['cy']
    matrix_camera[2, 2] = 1
    dist_coeffs = np.zeros((4, 1))

    if "dist_coeffs" in params["camera_settings"]:
        dist_coeffs = np.array(params["camera_settings"]['dist_coeffs'])

    config_detect = lambda: None
    config_detect.mask_edges = 1
    config_detect.mask_faces = 1
    config_detect.vertex = 1
    config_detect.threshold = 0.5
    config_detect.softmax = 1000
    config_detect.thresh_angle = params['thresh_angle']
    config_detect.thresh_map = params['thresh_map']
    config_detect.sigma = params['sigma']
    config_detect.thresh_points = params["thresh_points"]
    return matrix_camera, dist_coeffs, config_detect