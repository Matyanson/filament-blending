import itertools
import math
import numpy as np
from skimage import color
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, ConvexHull
import glfw

from glsl_helper import look_at, perspective
from helper import get_voxel_volume
from main_dispatcher import ShaderPipeline
from renderer import VolumeRenderer

# --- INPUT ---

filaments = {
    'Cotton White': {
        'rgb': [0.9019607843137255, 0.8666666666666667, 0.8588235294117647],
        'td': 0.41,   # Transmission Distance in mm
    },
    'Lava Red': {
        'rgb': [0.8705882352941177, 0.08627450980392157, 0.09803921568627451],
        'td': 0.38,
    },
    'Earth Brown': {
        'rgb': [0.48627450980392156, 0.34901960784313724, 0.2901960784313726],
        'td': 0.3,
    },
    'Sapphire Blue': {
        'rgb': [0.0, 0.35294117647058826, 0.6352941176470588],
        'td': 0.4,
    },
}

layer_thickness = 0.2  # mm per layer
max_filaments = 3


# --- SETUP ---

def alpha_from_td(thickness, td):
    """
    Convert thickness (mm) and transmission distance (mm) to an opacity:
      alpha = 1 - exp(-thickness / td)
    """
    return 1 - np.exp(-thickness / td)

filament_colors = []
base_points = []
base_points_alpha = []

for f_name, f in filaments.items():
    rgb = np.array(f['rgb'])
    td = f['td']
    alpha = alpha_from_td(layer_thickness, td)
    filament_colors.append({
        "key": f_name,
        "rgb": rgb,
        "alpha": alpha
    })
    base_points.append(rgb)
    base_points_alpha.append(alpha)
filament_colors = np.array(filament_colors)
base_points = np.array(base_points)
base_points_alpha = np.array(base_points_alpha)

# --- FUNCTIONS ---
def save_all_permutations(unique_indices):
    unique_indices = [int(fid) for fid in unique_indices]

    for filament_order in itertools.permutations(unique_indices):

        # 5) Find the optimal filament thickness for color blending (layers)
        layers = pipeline.run_blend_colors(filament_order, f"blend_unrestricted{filament_order}.png")

        # 6) modify layers to be compatible with 3d printing

        # 6.1: flatten 0th layer
        max0 = layers[:, :, 0].max()
        layers[:, :, 0] = max0

        # 6.2: smoothen layers that are not 0 or n-1
        offset = layers[:, :, 0].copy()
        num_layers = layers.shape[2]

        for i in range(1, num_layers - 1):
            layer = layers[:, :, i]
            elevation = offset + layer
            elevation_smooth = pipeline.run_smoothing(elevation)
            elevation_smooth = np.maximum(np.astype(elevation_smooth, np.int32), offset)
            layer_smooth = elevation_smooth - offset
            layers[:, :, i] = layer_smooth
            offset = elevation_smooth


        # FINAL: render the 3d volume!
        voxel_data, volume_dimensions = get_voxel_volume(filament_order, layers)

        pipeline.init_input()
        pipeline.set_volume(voxel_data)
        pipeline.save_volume_screenshot(f"blend{list(filament_order)}.png")

def get_filament_order(unique_indices):
    # filament_order = sorted(unique_indices, key=lambda index: filament_colors[index]['alpha'], reverse=True)
    # filament_order = sorted(unique_indices, key=lambda index: color_percentages[index])
    filament_error = []
    for fid in unique_indices:
        layers = pipeline.run_blend_colors([fid], "test_err.png")
        layer = layers[:, :, 0]

        layer_smooth = pipeline.run_smoothing(layer)
        layer_error = np.abs(layer - layer_smooth)
        error = np.sum(layer_error)
        filament_error.append(error)

    filament_order = sorted(enumerate(unique_indices), key=lambda pair: filament_error[pair[1]])
    filament_order = [fid for i, fid in filament_order]

    return filament_order

def smoothen_layers(layers):
    # 6.1: flatten 0th layer
    max0 = layers[:, :, 0].max()
    layers[:, :, 0] = max0

    # 6.2: smoothen layers that are not 0 or n-1
    offset = layers[:, :, 0].copy()
    num_layers = layers.shape[2]

    for i in range(1, num_layers - 1):
        layer = layers[:, :, i]
        elevation = offset + layer
        elevation_smooth = pipeline.run_smoothing(elevation)
        elevation_smooth = np.maximum(np.astype(elevation_smooth, np.int32), offset)
        layer_smooth = elevation_smooth - offset
        layers[:, :, i] = layer_smooth
        offset = elevation_smooth


# --- MAIN ---

# setup glsl context
win_w = 1000
win_h = 1000
glfw.init()
window = glfw.create_window(win_w, win_h, "Voxel Renderer", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)


# 1) CPU: calculate Delaunay
S = np.asarray(base_points)
delaunay = Delaunay(S)
hull = ConvexHull(S)

# 2) GPU: find optimal filament_colors

# input:    delaunay verticies (or rather the mesh), target_image
# output:   used delaunay vertices (indexes), result_image

# Prepare tetrahedrons and hull triangles
tets = delaunay.simplices  # shape (n, 4), each tet is a list of 4 vertex indices
hull_tris = hull.simplices  # shape (m, 3), each triangle is a list of 3 vertex indices

# 1) Setup shader pipeline
pipeline = ShaderPipeline(
    target_img_path='./input/wolf1.jpg',
    base_points=base_points,
    base_points_alpha=base_points_alpha
)

width, height = pipeline.get_texture_dimensions()

# 2) Mix colors
result_indices, result_coords = pipeline.run_mix_colors(tets, hull_tris)



# 3) Collect unique vertex indices used
unique_indices = set()
for row in result_indices.reshape(-1, 4):
    for idx in row:
        if idx >= 0:  # Skip unused (-1) entries
            unique_indices.add(idx)

unique_indices = list(unique_indices)

# 3.5) reduce number of filaments used
# get the color volume %
contributions = pipeline.run_get_color_contribution(unique_indices)

color_percentages = contributions.sum(axis=(0, 1))
color_percentages /= (width * height)

# unique_indices = sorted(unique_indices, key=lambda index: color_percentages[index], reverse=True)

# if(len(unique_indices) > max_filaments):
#     unique_indices = unique_indices[:max_filaments]


# 4) calculate filament order (most opaque at the bottom=0)

filament_order = get_filament_order(unique_indices)
filament_order = [int(fid) for fid in filament_order]


# 5) Find the optimal filament thickness for color blending (layers)
layers = pipeline.run_blend_colors(filament_order, f"blend_unrestricted{filament_order}.png")


# 6) modify layers to be compatible with 3d printing

smoothen_layers(layers)


# FINAL: render the 3d volume!
voxel_data, volume_dimensions = get_voxel_volume(filament_order, layers)

pipeline.init_input()
pipeline.set_volume(voxel_data)
pipeline.save_volume_screenshot(f"blend{list(filament_order)}.png")

W, H, D = volume_dimensions  # (512, 512, 29)
print("volume_dimensions: ", volume_dimensions)

camera_pos    = np.array([W/2, H/2, 300.0], dtype=np.float32)
renderer = VolumeRenderer(
    window,
    win_w,
    win_h,
    'shaders_render/fullscreen.vert',
    'shaders_render/raymarch.frag'
)
renderer.set_camera(camera_pos)
renderer.set_volume(0, voxel_data)

while not glfw.window_should_close(window):
    renderer.render_frame(step_size=0.1)

renderer.cleanup()
pipeline.cleanup()


print("filament_order: ", filament_order)
print("filament colors:", filament_colors)