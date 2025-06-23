import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram
from PIL import Image
import ctypes
import glfw

def get_glsl_format(dtype):
    # Mapping from numpy dtype to (internal_format, format, type)
    format_map = {
        np.uint8:  (GL_R8UI, GL_RED_INTEGER, GL_UNSIGNED_BYTE),
        np.uint16: (GL_R16UI, GL_RED_INTEGER, GL_UNSIGNED_SHORT),
        np.uint32: (GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT),
        np.int8:   (GL_R8I, GL_RED_INTEGER, GL_BYTE),
        np.int16:  (GL_R16I, GL_RED_INTEGER, GL_SHORT),
        np.int32:  (GL_R32I, GL_RED_INTEGER, GL_INT),
        np.float32:(GL_R32F, GL_RED, GL_FLOAT),  # For floating-point textures
    }
    # Normalize: make sure we’ve got a numpy.dtype
    dt = np.dtype(dtype)
    scalar_type = dt.type

    if scalar_type not in format_map:
        raise ValueError(f"Unsupported dtype: {scalar_type}")
    
    return format_map[scalar_type]

def bind_image_texture(path, binding_index, access=GL_READ_ONLY):    
    img = Image.open(path).convert('RGBA')
    img_data = np.array(img).astype(np.uint8)
    height, width = img_data.shape[:2]

    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, width, height)
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                    GL_RGBA, GL_UNSIGNED_BYTE, img_data)

    # Bind as image2D instead of sampler2D
    glBindImageTexture(binding_index, tex, 0, GL_FALSE, 0, access, GL_RGBA8)

    return width, height, tex

def bind_3d_texture(binding_index, data, access=GL_READ_ONLY):
    
    dtype = data.dtype
    internal_format, pixel_format, pixel_type = get_glsl_format(dtype)

    height, width, depth = data.shape
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_3D, tex)
    glTexImage3D(GL_TEXTURE_3D, 0, internal_format,
                 width, height, depth, 0,
                 pixel_format, pixel_type, data)
    glBindImageTexture(binding_index, tex, 0, GL_TRUE, 0, access, internal_format)

def base_align_data(data, dtype):
    arr = data.astype(dtype)
    # If a 2D array (array of vectors), ensure last axis is padded to multiple of 4
    if arr.ndim == 2:
        rows, cols = arr.shape
        # compute padded column count (multiple of 4)
        pad_cols = ((cols + 3) // 4) * 4
        if pad_cols != cols:
            # pad zeros on the right for each row
            pad_width = ((0, 0), (0, pad_cols - cols))
            arr = np.pad(arr, pad_width, mode='constant', constant_values=0)
    # else leave 1D or higher dims unchanged
    return arr

def create_ssbo(binding_index, data, dtype=np.float32):
    arr = base_align_data(data, dtype)

    buf_id = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf_id)
    glBufferData(GL_SHADER_STORAGE_BUFFER, arr.nbytes, arr, GL_STATIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_index, buf_id)
    return buf_id

def update_ssbo(buf_id, data, dtype):
    arr = base_align_data(data, dtype)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf_id)
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, arr.nbytes, arr)

def create_update_ssbo(buf_id, binding_index, data, dtype=np.float32):
    if(buf_id is None):
        return create_ssbo(binding_index, data, dtype)
    else:
        update_ssbo(buf_id, data, dtype)
    return buf_id


def load_compute_shader(path):
    with open(path, 'r') as f:
        source = f.read()
    return compileProgram(compileShader(source, GL_COMPUTE_SHADER))

def copy_shader_buffer(buffer_id, dtype, count):
    if dtype == np.int32:       ctype_arr = ctypes.c_int32 * count
    elif dtype == np.float32:   ctype_arr = ctypes.c_float * count
    else:                       raise ValueError(f"Unsupported dtype: {dtype}")
    
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer_id)
    ptr = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY)
    buffer_ptr = ctypes.cast(ptr, ctypes.POINTER(ctype_arr))
    result = np.frombuffer(buffer_ptr.contents, dtype=dtype).copy()
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)

    return result

def read_and_save_tex(tex, filename, w, h):
    glBindTexture(GL_TEXTURE_2D, tex)
    pixels = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE)
    arr = np.frombuffer(pixels, dtype=np.uint8).reshape((h, w, 4))
    Image.fromarray(arr, mode="RGBA").save(filename)



class ShaderPipeline:
    def __init__(self, target_img_path, base_points, base_points_alpha):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        self.window = glfw.create_window(1, 1, "", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)

        # Image input and output
        self.width, self.height, self.target_tex = bind_image_texture(target_img_path, 0, GL_READ_WRITE)

        self.num_pixels = self.width * self.height

        # Input SSBOs (persist across shaders)
        create_ssbo(1, np.array(base_points, dtype=np.float32), np.float32)
        create_ssbo(2, np.array(base_points_alpha, dtype=np.float32), np.float32)

        # Output buffers shared by both shaders
        self.output_indices = create_ssbo(5, np.zeros((self.num_pixels, 4), dtype=np.int32), np.int32)
        self.output_coords = create_ssbo(6, np.zeros((self.num_pixels, 4), dtype=np.float32), np.float32)

        # Will be set later
        self.tets_buf = None # 3
        self.hull_buf = None # 4
        self.filament_order_buf = None  # 7
        self.out_layers_buf = None      # 8

    def dispatch_shader(self, shader_path):
        shader = load_compute_shader(shader_path)
        glUseProgram(shader)
        glDispatchCompute((self.width + 15) // 16, (self.height + 15) // 16, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    def run_mix_colors(self, tets, hull_tris):
        # 0: target_tex
        # 1: base_points
        # 3: tets
        # 4: hull_tris
        # 5: out_indexes    - base point indexes
        # 6: out_bary       - barycentric coords

        # Input buffers
        self.tets_buf = create_update_ssbo(self.tets_buf, 3, np.array(tets, dtype=np.int32), np.int32)
        self.hull_buf = create_update_ssbo(self.hull_buf, 4, np.array(hull_tris, dtype=np.int32), np.int32)

        self.dispatch_shader('./mix_colors.comp')

        # Read back results
        indices = copy_shader_buffer(self.output_indices, np.int32, self.num_pixels * 4)
        coords = copy_shader_buffer(self.output_coords, np.float32, self.num_pixels * 4)

        indices = indices.reshape((self.height, self.width, 4))
        coords = coords.reshape((self.height, self.width, 4))

        read_and_save_tex(self.target_tex, "output/mixed.png", self.width, self.height)
        return indices, coords

    def run_blend_colors(self, filament_order):
        # 0: target_image
        # 1: base_points
        # 2: base_points_alpha
        # 3: filament_order
        # 4: out_layers         - thickness of each filament layer
        # 5: out_indexes
        # 6: out_bary

        # Input buffers
        self.filament_order_buf = create_update_ssbo(self.filament_order_buf, 3, np.array(filament_order, dtype=np.int32), np.int32)

        # Output buffers
        n = len(filament_order)
        self.out_layers_buf = create_update_ssbo(self.out_layers_buf, 4, np.zeros((self.num_pixels, n), dtype=np.int32), np.int32)

        self.dispatch_shader('./blend_colors.comp')
        read_and_save_tex(self.target_tex, "output/blended.png", self.width, self.height)

    def cleanup(self):
        glfw.terminate()
