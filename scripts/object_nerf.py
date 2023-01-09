import bpy
import os
import sys
dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append(dir)
from scripts.common import *


if __name__ == '__main__':
    args = get_args()

    scene = bpy.context.scene
    world = scene.world
    camera = scene.camera

    obj_name = import_obj(args.object)
    fit_to_origin_box(obj_name, BOX_SIZE)

    # Create render background
    node_tree = world.node_tree
    node_tree.nodes.clear()

    node_background = node_tree.nodes.new(type='ShaderNodeBackground')
    node_output = node_tree.nodes.new(type='ShaderNodeOutputWorld')
    node_tree.links.new(node_background.outputs["Background"], node_output.inputs["Surface"])

    # constant white illumination
    node_tree.nodes["Background"].inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)

    # Make background invisible to camera. Only keeps lighting
    world.cycles_visibility.camera = False
    scene.render.film_transparent = True

    # Color mapping and format setting
    scene.view_settings.view_transform = args.color
    scene.render.image_settings.file_format = args.format

    # setup directories
    output_dir = os.path.join(os.getcwd(), args.output)
    img_dir = os.path.join(output_dir, IMG_FOLDER_NAME)
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    scene.render.filepath = img_dir

    # render
    frames = render_nerf_dataset(scene, camera, args.views, args.mode)

    # write transform.json
    data = get_camera_intrinsics(scene, camera)
    data['frames'] = frames
    filepath = os.path.join(args.output, JSON_NAME)
    save_json(filepath, data)
