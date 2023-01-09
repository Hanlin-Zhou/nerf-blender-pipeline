import bpy
import sys
import os
import argparse
import json
import numpy as np
import math

# bounding box of size 3.0
# https://github.com/NVlabs/instant-ngp/blob/master/docs/nerf_dataset_tips.md#existing-datasets
BOX_SIZE = 3.0
AABB_SCALE = 16
SPHERE_RADIUS = 7.0           # in 'random_hemisphere', 'random_sphere', and 'circular' trajectory
CAMERA_TRACK_HEIGHT = 2.0     # in 'circular' and 'figure_eight' trajectory
FIGURE_EIGHT_SCALE = 5.0      # in 'figure_eight' trajectory
JSON_NAME = 'transforms.json'
IMG_FOLDER_NAME = './images'


def get_args(composite=False):
    try:
        idx = sys.argv.index("--")
        script_args = sys.argv[idx + 1:]
    except ValueError as e:  # '--' not in the list:
        script_args = []

    parser = argparse.ArgumentParser()
    parser.add_argument('--object', type=str, help='path to foreground object', required=True)
    parser.add_argument('--hdri', type=str, help='path to HDRI', required=composite)
    parser.add_argument('--output', type=str, help='path to generated dataset', default="./")
    parser.add_argument('--views', type=int, help='number of views to render', default=30)
    parser.add_argument('--mode', type=str, help='render mode ("random", "circular")', default="random_hemisphere",
                        choices=['random_hemisphere', 'random_sphere', 'circular', 'figure_eight'])
    parser.add_argument('--format', type=str, help='output file format', default="PNG",
                        choices=['BMP', 'IRIS', 'PNG', 'JPEG', 'JPEG2000', 'TARGA', 'TARGA_RAW', 'CINEON', 'DPX', 'HDR',
                                 'OPEN_EXR_MULTILAYER', 'OPEN_EXR', 'TIFF', 'WEBP', 'AVI_JPEG', 'AVI_RAW', 'FFMPEG'])
    parser.add_argument('--color', type=str, help='View Transform of render', default="Standard",
                        choices=['Standard', 'Filmic', 'Filmic Log', 'Raw', 'False Color'])

    args = parser.parse_args(script_args)
    return args


def import_obj(path):
    """
        Import .obj file. Terminate if file not found.
        Return reference name of imported model
    """
    if os.path.exists(path):
        bpy.ops.import_scene.obj(filepath=path, global_clamp_size=0.0)
        bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
        bpy.ops.object.join()
        obj_object = bpy.context.selected_objects[0]
        return obj_object.name
    else:
        print(f"file {path} does not exist. Check path.")
        bpy.ops.wm.quit_blender()
        return None


def import_hdri(path):
    """
        Import HDRI file. Terminate if file not found.
        Return reference name of imported model
    """
    filepath = os.path.join(os.getcwd(), path)
    if os.path.exists(filepath):
        return bpy.data.images.load(filepath=filepath)


def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list


def save_json(filepath, data):
    with open(filepath, 'w') as file:
        json.dump(data, file, indent=4)


# https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
def get_camera_intrinsics(scene, camera, composite=False):
    camera_angle_x = camera.data.angle_x
    camera_angle_y = camera.data.angle_y

    f_in_mm = camera.data.lens
    scale = scene.render.resolution_percentage / 100
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    width_res_in_px = scene.render.resolution_x * scale
    height_res_in_px = scene.render.resolution_y * scale

    optical_center_x = width_res_in_px / 2
    optical_center_y = height_res_in_px / 2

    sensor_size_in_mm = camera.data.sensor_height if camera.data.sensor_fit == 'VERTICAL' else camera.data.sensor_width

    size_x = scene.render.pixel_aspect_x * width_res_in_px
    size_y = scene.render.pixel_aspect_y * height_res_in_px

    if camera.data.sensor_fit == 'AUTO':
        sensor_fit = 'HORIZONTAL' if size_x >= size_y else 'VERTICAL'
    else :
        sensor_fit = camera.data.sensor_fit

    view_fac_in_px = width_res_in_px if sensor_fit == 'HORIZONTAL' else pixel_aspect_ratio * height_res_in_px
    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px

    s_u = 1 / pixel_size_mm_per_px
    s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

    camera_intr_dict = {
        'camera_angle_x': camera_angle_x,
        'camera_angle_y': camera_angle_y,
        'fl_x': s_u,
        'fl_y': s_v,
        'k1': 0.0,
        'k2': 0.0,
        'p1': 0.0,
        'p2': 0.0,
        'cx': optical_center_x,
        'cy': optical_center_y,
        'w': width_res_in_px,
        'h': height_res_in_px,
    }

    if composite:
        camera_intr_dict["aabb_scale"] = AABB_SCALE

    return camera_intr_dict


def fit_to_origin_box(obj_name, box_size):
    """
        Translate the object to origin and scale proportional to fit inside bounding box of side length box_size
        Return transformed Bound_Box as numpy array with Z-up
    """
    obj = bpy.data.objects[obj_name]
    obj_mat = np.array(obj.bound_box)
    # Y-up to Z-up
    obj_mat[:, [2, 1]] = obj_mat[:, [1, 2]]
    obj_center = (np.amax(obj_mat, axis=0) + np.amin(obj_mat, axis=0)) / 2.0
    obj_size = np.amax(obj_mat, axis=0) - np.amin(obj_mat, axis=0)
    scale_factor = box_size / np.amax(obj_size)
    bpy.ops.transform.translate(value=-obj_center)
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
    bpy.ops.transform.resize(value=(scale_factor, scale_factor, scale_factor))
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=True)

    mat = np.array(bpy.data.objects[obj_name].bound_box)
    mat[:, [2, 1]] = mat[:, [1, 2]]
    return mat


def render_nerf_dataset(scene, camera, num_views, mode):
    if mode == "random_hemisphere":
        return _render_uniform_sphere(scene, camera, num_views, True)
    elif mode == "random_sphere":
        return _render_uniform_sphere(scene, camera, num_views, False)
    elif mode == "circular":
        return _render_circular(scene, camera, num_views)
    elif mode == "figure_eight":
        return _render_figure_eight(scene, camera, num_views)


def camera_look_at(scene, camera, camera_init_coord=(0.0, 0.0, 0.0), coord_look_at=(0.0, 0.0, 0.0), make_parent=True):
    obj_look_at = bpy.data.objects.new('LookAt', None)
    obj_look_at.location = coord_look_at
    scene.collection.objects.link(obj_look_at)
    camera_lookat = camera.constraints.new(type='TRACK_TO')
    camera_lookat.target = obj_look_at
    camera.location = camera_init_coord
    if make_parent:
        camera.parent = obj_look_at
    return obj_look_at


def _render_uniform_sphere(scene, camera, num_views, only_hemisphere):
    """
        Uniformly sample views on surface of sphere/hemisphere
    """
    origin = camera_look_at(scene, camera, (SPHERE_RADIUS, 0, 0))

    frames = []
    img_dir = scene.render.filepath
    random2d = np.random.uniform(size=(num_views, 2))
    for i in range(num_views):
        theta = 2.0 * math.pi * random2d[i, 0]
        if only_hemisphere:
            phi = math.acos(random2d[i, 1]) - math.pi / 2.0
        else:
            phi = math.acos(1.0 - 2.0 * random2d[i, 1]) - math.pi / 2.0

        origin.rotation_euler[2] = theta
        origin.rotation_euler[1] = phi

        scene.render.filepath = os.path.join(img_dir, str(i))
        bpy.ops.render.render(write_still=True)

        # Frames info
        frame_data = {
                'file_path': os.path.join(IMG_FOLDER_NAME, str(i)),
                'transform_matrix': listify_matrix(camera.matrix_world)
        }
        frames.append(frame_data)

    return frames


def _render_circular(scene, camera, num_views):
    """
        Sample view in circular track
    """
    origin = camera_look_at(scene, camera, (SPHERE_RADIUS, 0, CAMERA_TRACK_HEIGHT))

    frames = []
    img_dir = scene.render.filepath

    step_size = 2.0 * math.pi / num_views
    for i in range(num_views):
        theta = step_size * i
        origin.rotation_euler[2] = theta

        scene.render.filepath = os.path.join(img_dir, str(i))
        bpy.ops.render.render(write_still=True)

        # Frames info
        frame_data = {
            'file_path': os.path.join(IMG_FOLDER_NAME, str(i)),
            'transform_matrix': listify_matrix(camera.matrix_world)
        }
        frames.append(frame_data)

    return frames


def _render_figure_eight(scene, camera, num_views):
    """
        Sample views in shape of number 8 (Lemniscate of Gerono)
    """
    origin = camera_look_at(scene, camera,make_parent=False)

    frames = []
    img_dir = scene.render.filepath

    step_size = 2.0 * math.pi / num_views
    for i in range(num_views):
        theta = step_size * i
        _x = FIGURE_EIGHT_SCALE * math.cos(theta) + 1e-2  # avoid camera straight down
        _y = FIGURE_EIGHT_SCALE * math.sin(2.0 * theta)
        camera.location = (_x, _y, CAMERA_TRACK_HEIGHT)

        scene.render.filepath = os.path.join(img_dir, str(i))
        bpy.ops.render.render(write_still=True)

        # Frames info
        frame_data = {
            'file_path': os.path.join(IMG_FOLDER_NAME, str(i)),
            'transform_matrix': listify_matrix(camera.matrix_world)
        }
        frames.append(frame_data)

    return frames
