import math
import numpy as np
from skimage import color
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, ConvexHull
import glfw

from glsl_helper import look_at, perspective
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
target_colors = np.vstack(([0.3, 0.7, 0.5], np.random.rand(11, 3)))


# --- SETUP ---

def alpha_from_td(thickness, td):
    """
    Convert thickness (mm) and transmission distance (mm) to an opacity:
      alpha = 1 - exp(-thickness / td)
    """
    return 1 - np.exp(-thickness / td)

target_image = Image.open('./input/target_200.png').convert('RGB')
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




# --- MAIN ---

# setup glsl context
win_w = 500
win_h = 500
glfw.init()
window = glfw.create_window(win_w, win_h, "Voxel Renderer", None, None)
glfw.make_context_current(window)


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

# Setup shader pipeline
pipeline = ShaderPipeline(
    target_img_path='./input/target_512.png',
    base_points=base_points,
    base_points_alpha=base_points_alpha
)

width, height = pipeline.get_texture_dimensions()

# Mix colors
result_indices, result_coords = pipeline.run_mix_colors(tets, hull_tris)



# 3) Collect unique vertex indices used
unique_indices = set()
for row in result_indices.reshape(-1, 4):
    for idx in row:
        if idx >= 0:  # Skip unused (-1) entries
            unique_indices.add(idx)

# 4) calculate filament order
unique_indices = list(unique_indices)
filament_order = sorted(unique_indices, key=lambda index: filament_colors[index]['alpha'])

pipeline.run_blend_colors(filament_order)

voxel_tex, volume_dimensions = pipeline.run_raymarching(filament_order)

W, H, D = volume_dimensions  # (512, 512, 29)
center = np.array([-W/8, H/2, D/2], dtype=np.float32)  # (256,256,14.5)
eye    = np.array([center[0], center[1], -100.0], dtype=np.float32)
up     = np.array([0, 0, 1], dtype=np.float32)
view_mat = look_at(eye, center, up)

# projection
fovy    = np.radians(45.0)
aspect  = W / H
znear   = 0.1
zfar    = np.linalg.norm([W, H, D]) * 3.0

proj_mat = perspective(fovy, aspect, znear, zfar)

renderer = VolumeRenderer(
    window,
    win_w,
    win_h,
    'fullscreen.vert',
    'raymarch.frag'
)
print("volume_dimensions: ", volume_dimensions)
renderer.set_volume(voxel_tex, volume_dimensions)
renderer.set_camera(view_mat, proj_mat)

while not glfw.window_should_close(window):
    renderer.render_frame(step_size=0.1)

renderer.cleanup()
pipeline.cleanup()


print("filament_order: ", filament_order)
print("Unique filament indices used:", unique_indices)

print(len(base_points), len(unique_indices))