import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram
from PIL import Image
import ctypes
import glfw

def get_glsl_format(dtype):
    # Mapping from numpy dtype to (internal_format, pixel_format, pixel_type)
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

def bind_image_texture_from_path(path, binding_index, access=GL_READ_ONLY):    
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

def save_texture(img_data, dtype):
    internal_format, pixel_format, pixel_type = get_glsl_format(dtype)

    H, W = img_data.shape[:2]
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexStorage2D(GL_TEXTURE_2D, 1, internal_format, W, H)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H,
                    pixel_format, pixel_type, img_data)
    return tex

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

    return tex

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
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    return compileProgram(compileShader(source, GL_COMPUTE_SHADER))

def copy_shader_buffer(buffer_id, dtype, count):
    if dtype == np.int32:       ctype_arr = ctypes.c_int32 * count
    elif dtype == np.uint32:       ctype_arr = ctypes.c_uint32 * count
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

def read_tex(tex_id, w, h, dtype):
    internal_format, pixel_format, pixel_type = get_glsl_format(dtype)

    glBindTexture(GL_TEXTURE_2D, tex_id)
    raw = glGetTexImage(GL_TEXTURE_2D, 0, pixel_format, pixel_type)
    arr = np.frombuffer(raw, dtype).reshape(h, w)

    return arr



class ShaderPipeline:
    def __init__(self, target_img_path, base_points, base_points_alpha):

        # glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        # Image input and output
        self.width, self.height, self.target_tex = bind_image_texture_from_path(target_img_path, 0, GL_READ_WRITE)

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
        self.filament_order_buf = None  # 3
        self.out_layers_buf = None      # 4

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

        self.dispatch_shader('shaders_compute/mix_colors.comp')

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
        self.out_layers_buf = create_update_ssbo(self.out_layers_buf, 4, np.zeros(self.num_pixels * n, dtype=np.int32), np.int32)

        self.dispatch_shader('shaders_compute/blend_colors.comp')
        read_and_save_tex(self.target_tex, "output/blended.png", self.width, self.height)

        # Read back the SSBO into a (W*H*N,) array
        flat = copy_shader_buffer(self.out_layers_buf, np.int32, self.num_pixels * n)
        layers = flat.reshape((self.height, self.width, n))
        return layers
    
    def run_layer_envelope(self, slice, mode):
        # 0: uA         - 2d slice with layer thickness values
        # 1: uPrev      - copy of the slice
        # 2: uNext      - copy of the slice (output)
        # 3: uMode      - calculate lower or upper envelope
        H, W = slice.shape
        prog = load_compute_shader("shaders_compute/morph.comp")
        glUseProgram(prog)

        # textures
        texA = save_texture(slice.copy(), np.float32)
        texPrev = save_texture(slice.copy(), np.float32)
        texNext = save_texture(slice.copy(), np.float32)

        # uniforms
        loc_mode = glGetUniformLocation(prog, "uMode")
        loc_dist_method = glGetUniformLocation(prog, "uDistMethod")
        loc_max_gradient = glGetUniformLocation(prog, "uMaxGradient")
        flag_ssbo = create_ssbo(3, np.array([0], np.uint32), np.uint32)

        # set uniforms
        glUniform1i(loc_mode, mode)
        glUniform1i(loc_dist_method, 0)
        glUniform1f(loc_max_gradient, 1.0)
        glBindImageTexture(0, texA,    0, GL_FALSE, 0, GL_READ_ONLY,  GL_R32F)

        for i in range(10000):
            # bind images
            glBindImageTexture(1, texPrev, 0, GL_FALSE, 0, GL_READ_ONLY,  GL_R32F)
            glBindImageTexture(2, texNext, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F)

            # dispatch
            gx = (W + 15) // 16
            gy = (H + 15) // 16
            glDispatchCompute(gx, gy, 1)
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT)

            # swap ping-pong
            texPrev, texNext = texNext, texPrev

            # check for changes
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, flag_ssbo)
            flag_value = copy_shader_buffer(flag_ssbo, np.uint32, 1)

            if flag_value[0] == 0:
                print(f"converged after {i} passes")
                break

            # clear the flag
            zero = np.array(0, dtype=np.uint32)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, flag_ssbo)
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 4, zero)

        # read the result
        arr = read_tex(texNext, W, H, np.float32)

        return arr
    
    def run_average(self, arr1, arr2):
        # 0: uInput1
        # 1: uInput2
        H, W = arr1.shape
        prog = load_compute_shader("shaders_compute/average.comp")
        glUseProgram(prog)
        
        # textures
        texA    = save_texture(arr1, np.float32)
        texB    = save_texture(arr2, np.float32)

        # uniforms
        loc_mode = glGetUniformLocation(prog, "uMode")

        # set uniforms
        glUniform1i(loc_mode, 0)
        glBindImageTexture(0, texA,    0, GL_FALSE, 0, GL_READ_ONLY,  GL_R32F)
        glBindImageTexture(1, texB,    0, GL_FALSE, 0, GL_READ_ONLY,  GL_R32F)

        # dispatch
        gx = (W + 15) // 16
        gy = (H + 15) // 16
        glDispatchCompute(gx, gy, 1)
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

        # read the result
        arr = read_tex(texA, W, H, np.float32)

        return arr

    def run_smoothing(self, slice):
        lower_arr = self.run_layer_envelope(slice, 0)
        upper_arr = self.run_layer_envelope(slice, 0)

        average_arr = self.run_average(lower_arr, upper_arr)
        return average_arr





    def get_texture_dimensions(self):
        return self.width, self.height

    def cleanup(self):
        glfw.terminate()
