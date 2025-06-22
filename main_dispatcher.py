import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram
from PIL import Image
import ctypes
import glfw


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


def create_ssbo(data, binding_index, dtype=np.float32):
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

    buf = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf)
    glBufferData(GL_SHADER_STORAGE_BUFFER, arr.nbytes, arr, GL_STATIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_index, buf)
    return buf


def load_compute_shader(path):
    with open(path, 'r') as f:
        source = f.read()
    return compileProgram(compileShader(source, GL_COMPUTE_SHADER))

def copy_shader_buffer(buffer_id, dtype, count):
    if dtype == np.int32:
        ctype_arr = ctypes.c_int32 * count
    elif dtype == np.float32:
        ctype_arr = ctypes.c_float * count
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
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
        create_ssbo(np.array(base_points, dtype=np.float32), 1, np.float32)
        create_ssbo(np.array(base_points_alpha, dtype=np.float32), 2, np.float32)

        # Output buffers shared by both shaders
        self.output_indices = create_ssbo(np.zeros((self.num_pixels, 4), dtype=np.int32), 5, np.int32)
        self.output_coords = create_ssbo(np.zeros((self.num_pixels, 4), dtype=np.float32), 6, np.float32)

        # Will be set later
        self.filament_order_buf = None  # 7
        self.out_layers_buf = None      # 8

    def dispatch_shader(self, shader_path):
        shader = load_compute_shader(shader_path)
        glUseProgram(shader)
        glDispatchCompute((self.width + 15) // 16, (self.height + 15) // 16, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    def run_mix_colors(self, tets, hull_tris):

        # Input buffers
        create_ssbo(np.array(tets, dtype=np.int32), 3, np.int32)
        create_ssbo(np.array(hull_tris, dtype=np.int32), 4, np.int32)

        self.dispatch_shader('./mix_colors.comp')

        # Read back results
        indices = copy_shader_buffer(self.output_indices, np.int32, self.num_pixels * 4)
        coords = copy_shader_buffer(self.output_coords, np.float32, self.num_pixels * 4)

        indices = indices.reshape((self.height, self.width, 4))
        coords = coords.reshape((self.height, self.width, 4))

        read_and_save_tex(self.target_tex, "output/mixed.png", self.width, self.height)
        return indices, coords

    def run_blend_colors(self, filament_order):

        # Input buffers
        self.filament_order_buf = create_ssbo(np.array(filament_order, dtype=np.int32), 7, dtype=np.int32)

        # Output buffers
        n = len(filament_order)
        self.out_layers_buf = create_ssbo(np.zeros((self.num_pixels, n), dtype=np.int32), 8, dtype=np.int32)

        self.dispatch_shader('./blend_colors.comp')
        read_and_save_tex(self.target_tex, "output/blended.png", self.width, self.height)

    def cleanup(self):
        glfw.terminate()
