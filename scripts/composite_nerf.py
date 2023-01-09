import bpy
import os
import sys
dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append(dir)
from scripts.common import *


if __name__ == '__main__':
    args = get_args(composite=True)

    scene = bpy.context.scene
    world = scene.world
    camera = scene.camera

    # Load HDRI
    hdri = import_hdri(args.hdri)

    # Load .obj
    obj_name = import_obj(args.object)
    obj_bound_box = fit_to_origin_box(obj_name, BOX_SIZE)

    # add a plane right under the object
    obj_elevation = np.amin(obj_bound_box, axis=0)[2]
    bpy.ops.mesh.primitive_plane_add(size=BOX_SIZE, enter_editmode=False, align='WORLD',
                                     location=(0, 0, obj_elevation), scale=(1, 1, 1))

    # Create render background
    node_tree = world.node_tree
    node_tree.nodes.clear()

    node_background = node_tree.nodes.new(type='ShaderNodeBackground')
    node_environment = node_tree.nodes.new(type='ShaderNodeTexEnvironment')
    node_output = node_tree.nodes.new(type='ShaderNodeOutputWorld')
    node_environment.image = hdri
    node_tree.links.new(node_environment.outputs["Color"], node_background.inputs["Color"])
    node_tree.links.new(node_background.outputs["Background"], node_output.inputs["Surface"])

    # Make background visible
    world.cycles_visibility.camera = True
    scene.render.film_transparent = False

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
    data = get_camera_intrinsics(scene, camera, composite=True)
    data['frames'] = frames
    filepath = os.path.join(args.output, JSON_NAME)
    save_json(filepath, data)
