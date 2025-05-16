import cv2
import magnum as mn
import numpy as np
from habitat_sim.utils import common as utils
import habitat_sim
import open3d as o3d


def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings['scene']
    sim_cfg.scene_dataset_config_file = settings['scene_dataset']
    sim_cfg.enable_physics = settings["enable_physics"]
    # Note: all sensors must have the same resolution
    sensor_specs = []

    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = 'color_sensor'
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [settings['height'], settings['width']]
    color_sensor_spec.position = [0.0, settings['camera_height'], 0.0]
    color_sensor_spec.hfov = settings['hfov']
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(color_sensor_spec)

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = 'depth_sensor'
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings['height'], settings['width']]
    depth_sensor_spec.position = [0.0, settings['camera_height'], 0.0]
    depth_sensor_spec.hfov = settings['hfov']
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(depth_sensor_spec)

    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = 'semantic_sensor'
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings['height'], settings['width']]
    semantic_sensor_spec.position = [0.0, settings['camera_height'], 0.0]
    semantic_sensor_spec.hfov = settings['hfov']
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(semantic_sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        'turn_right_1degree': habitat_sim.agent.ActionSpec(
            'turn_right', habitat_sim.agent.ActuationSpec(amount=1)),
        'turn_right_15degree': habitat_sim.agent.ActionSpec(
            'turn_right', habitat_sim.agent.ActuationSpec(amount=15)),
        'turn_left_15degree': habitat_sim.agent.ActionSpec(
            'turn_left', habitat_sim.agent.ActuationSpec(amount=15)),
        'move_forward_0.25m': habitat_sim.agent.ActionSpec(
            'move_forward', habitat_sim.agent.ActuationSpec(amount=0.25))
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def get_camera_intrinsics(sim, sensor_name):
    # get render camera
    render_camera = sim._sensors[sensor_name]._sensor_object.render_camera
    # get projection matrix
    projection_matrix = render_camera.projection_matrix
    # get resolution
    viewport_size = render_camera.viewport
    # intrinsic calculation
    fx = projection_matrix[0, 0] * viewport_size[0] / 2.0
    fy = projection_matrix[1, 1] * viewport_size[1] / 2.0
    cx = (projection_matrix[2, 0] + 1.0) * viewport_size[0] / 2.0
    cy = (projection_matrix[2, 1] + 1.0) * viewport_size[1] / 2.0
    return fx, fy, cx, cy


def warp_image_by_homography(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)[:100]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC)

    aligned_img1 = cv2.warpPerspective(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY),
                                       H, (img2.shape[1], img2.shape[0]))

    return aligned_img1, cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


def generate_event_image_warped(img1, img2, threshold=5, use_log=True):
    aligned_gray1, gray2 = warp_image_by_homography(img1, img2)

    aligned_gray1 = aligned_gray1.astype(np.float32)
    gray2 = gray2.astype(np.float32)

    if use_log:
        aligned_gray1 = np.log1p(aligned_gray1)
        gray2 = np.log1p(gray2)

    diff = gray2 - aligned_gray1

    pos_event = (diff > threshold).astype(np.uint8)
    neg_event = (diff < -threshold).astype(np.uint8)

    event_img = np.zeros((*gray2.shape, 3), dtype=np.uint8)
    event_img[..., 1] = pos_event * 255  # Green
    event_img[..., 2] = neg_event * 255  # Red

    return event_img


if __name__ == "__main__":

    scene_name = 'D7N2EKCX4Sj'
    sim_settings = {
        "enable_physics": True,
        'scene': f'your/path/{scene_name}/{scene_name}.glb',  # Scene
        'scene_dataset': 'your/path/MP3D/mp3d.scene_dataset_config.json',  # the scene dataset configuration files
        'width': 512,  # Resolution of the observations
        'height': 512,  # Resolution of the observations
        'hfov': 90,  # Horizontal field of view
        'camera_height': 1.5,  # Camera height
    }

    # initialize an environment
    cfg = make_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)

    # initialize an agent
    agent = sim.initialize_agent(0)
    agent_state = habitat_sim.AgentState()
    pt1 = sim.pathfinder.get_random_navigable_point()
    pt2 = sim.pathfinder.get_random_navigable_point()
    tangent_orientation_matrix = mn.Matrix4.look_at(eye=pt1, target=pt2, up=np.array([0.0, 1.0, 0]))
    tangent_orientation_q = mn.Quaternion.from_matrix(tangent_orientation_matrix.rotation())
    agent_state.position = pt1
    agent_state.rotation = utils.quat_from_magnum(tangent_orientation_q)
    agent.set_state(agent_state)

    # get the current observations
    observations = sim.get_sensor_observations()
    rgb, semantic, depth = observations['color_sensor'], observations['semantic_sensor'], observations['depth_sensor']
    rgb_t1 = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)  # RGB to BGR

    # get point cloud from depth image
    fx, fy, cx, cy = get_camera_intrinsics(sim, 'color_sensor')
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width=sim_settings['width'], height=sim_settings['height'], fx=fx, fy=fy, cx=cx, cy=cy)
    extrinsic = tangent_orientation_matrix
    pcd = np.asarray(o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(depth), intrinsic=intrinsic, extrinsic=extrinsic, depth_scale=1.0, depth_trunc=50.0))

    # get event image
    sim.step('turn_right_1degree')
    observations = sim.get_sensor_observations()
    rgb, semantic, depth = observations['color_sensor'], observations['semantic_sensor'], observations['depth_sensor']
    rgb_t2 = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
    event_img = generate_event_image_warped(rgb_t1, rgb_t2, threshold=0.05, use_log=True)
    event_img = cv2.cvtColor(event_img, cv2.COLOR_BGR2RGB)