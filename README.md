# relu_fields
**Official** implementation of the paper: 
[**ReLU Fields: The Little Non-linearity That Could.**](https://geometry.cs.ucl.ac.uk/group_website/projects/2022/relu_fields/)
The implementation uses the **PyTorch** framework.

![GitHub](https://img.shields.io/github/license/akanimax/relu_fields)
![Generic badge](https://img.shields.io/badge/conf-SIGGRAPH2022-purple.svg)

![teaser](https://geometry.cs.ucl.ac.uk/group_website/projects/2022/relu_fields/static/figures/teaser_relufield_correct_font.jpg)

# Mitsuba 3 tutorial
Check out this 
[**Mitsuba 3**](https://mitsuba.readthedocs.io/en/latest/src/inverse_rendering/radiance_field_reconstruction.html) 
inverse rendering tutorial based on ReLU Fields.

# Radiance Fields Experiments
![hotdog](https://geometry.cs.ucl.ac.uk/group_website/projects/2022/relu_fields/static/videos/hotdog_spiral.gif)
### Data:
The data used for these experiments is available at this 
[**drive**](https://drive.google.com/drive/u/0/folders/1-iJug5cTJA7bhDnhIxTraH5EyuyRA7sr) 
link. Download the `synthetic_radiance_fields_dataset.zip` and extract it in some
folder on your disk. For easy access create a `/data` folder in this **repo-root** and
extract the zip in something like `/data/radiance_fields`.

### Run the optimization:
Running the optimization is super simple :smile:. Just use the script 
`train_sh_based_voxel_grid_with_posed_images.py` python script with appropriate
cmd-args. Following example runs the training on the **Hotdog** scene:

```
(relu_fields_env) user@machine:<repo_path>$ python train_sh_based_voxel_grid_with_posed_images.py -d data/radiance_fields/hotdog -o logs/rf/hotdog/
```

### Render trained model:
Use the `render_sh_based_voxel_grid.py` script for creating a rotating/spiral 
3D render of the trained models. Following are all the options that you can tweak
in this render_script
```
Usage: render_sh_based_voxel_grid.py [OPTIONS]

Options:
  -i, --model_path FILE           path to the trained (reconstructed) model
                                  [required]
  -o, --output_path DIRECTORY     path for saving rendered output  [required]
  --overridden_num_samples_per_ray INTEGER RANGE
                                  overridden (increased) num_samples_per_ray
                                  for beautiful renders :)  [x>=1]
  --render_scale_factor FLOAT     overridden (increased) resolution (again :D)
                                  for beautiful renders :)
  --camera_path [thre360|spiral]  which camera path to use for rendering the
                                  animation
  --camera_pitch FLOAT            pitch-angle value for the camera for 360
                                  path animation
  --num_frames INTEGER RANGE      number of frames in the video  [x>=1]
  --vertical_camera_height FLOAT  height at which the camera spiralling will
                                  happen
  --num_spiral_rounds INTEGER RANGE
                                  number of rounds made while transitioning
                                  between spiral radii  [x>=1]
  --fps INTEGER RANGE             frames per second of the video  [x>=1]
  --help                          Show this message and exit.

```

# Coming soon ...
The **Geometry (occupancy)**, **Pixel Fields (Images)** and the 
**Real-scene** experiments will be setup in the `thre3d_atom` package soon.

# BibTex:
```
@article{Karnewar2022ReLUFields,
    author    = {Karnewar, Animesh and Ritschel, Tobias and Wang, Oliver and Mitra, Niloy J.},
    title     = {ReLU Fields: The Little Non-linearity That Could},
    journal   = {Transactions on Graphics (Proceedings of SIGGRAPH),
    volume    = {41},
    number    = {4},
    year      = {2022},
    month     = july,
    pages     = {13:1--13:8},
    doi       = {10.1145/3528233.3530707},
}
```

# Thanks
As always, <br>
please feel free to open PRs/issues/suggestions here :smile:. 

cheers :beers:! <br>
@akanimax :sunglasses: :robot:
