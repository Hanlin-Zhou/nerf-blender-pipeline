# Blender NeRF Dataset Pipeline
Blender scripts for NeRF dataset generation.

Camera Trajectory Mode
------------
* `random_hemisphere` : uniformly sample the surface of a hemisphere looking at the object.
* `random_sphere` : uniformly sample the surface of a sphere looking at the object.
* `circular` : evenly sampling a circular path around the object.
* `figure_eight`: evenly sampling a figure 8 path with object at the center.



<p align='left'>
  <img src="assets\random_hemisphere.png" width="400" height="380"/>
  <img src="assets\random_sphere.png" width="400" height="380"/>
  <br>
  <img src="assets\circular.png" width="400" height="280"/>
  <img src="assets\figure_eight.png" width="400" height="280"/>
</p>

Usage
------------
__Tested with Blender 3.3__

`object_nerf.py`: Creates dataset with only the .obj model. Model is illuminated by white background lighting.

`composite_nerf.py`: Creates dataset with the .obj model sitting on a plane. Scene is illuminated by the HDRI specified.

`config.blend`: Scripts will use the render configurations in this empty blender file. 
Settings you might want to change:
* Rendering Device and backend
* Output image resolution. Default 800 x 800
* Camera focal length
* Render setting such as SPP, Max Bounces, Depth of Field. 

### Example: Object
<img src="assets\object_around.gif" />

```sh
nerf-pipeline> blender --background config.blend --python scripts\object_nerf.py -- --object models\bench3\model.obj --views 50 --mode random_sphere --output dataset\bench3
```
__Depending on your setup you might have to enter the full path to the Blender executable.__
### Example: Composite 
<img src="assets\composite_around.gif"/>

```sh
nerf-pipeline> blender --background config.blend --python scripts\composite_nerf.py -- --object models\bench1\model.obj --views 60 --mode random_hemisphere --output dataset\bench_hdri --hdri hdri\daylight.exr
```

