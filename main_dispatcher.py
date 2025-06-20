import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram
from PIL import Image
import ctypes
import glfw


def bind_image_texture(path, binding_index, dtype, access=GL_READ_ONLY):    
    img = Image.open(path).convert('RGBA')
    img_data = np.array(img).astype(np.uint8)
    print("input image pixels: \n", img_data)
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
    buf = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf)
    glBufferData(GL_SHADER_STORAGE_BUFFER, data.astype(dtype).nbytes, data, GL_STATIC_DRAW)
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
    print("output image pixels: \n", arr)
    Image.fromarray(arr, mode="RGBA").save(filename)


def run_shader(target_img_path, base_points, tets, hull_tris, shader_path):
    # Setup OpenGL context using GLFW
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(1, 1, "", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")

    glfw.make_context_current(window)

    shader = load_compute_shader(shader_path)

    width, height, target_tex = bind_image_texture(target_img_path, 0, GL_READ_WRITE)

    # Prepare input buffers
    create_ssbo(np.array(base_points, dtype=np.float32), 1, np.float32)
    create_ssbo(np.array(tets, dtype=np.int32), 2, dtype=np.int32)
    create_ssbo(np.array(hull_tris, dtype=np.int32), 3, dtype=np.int32)

    # Prepare output buffers
    num_pixels = width * height
    data_template = np.zeros((num_pixels, 4), dtype=np.int32)
    output_indices = create_ssbo(data_template, 4, dtype=np.int32)

    data_template = np.zeros((num_pixels, 4), dtype=np.float32)
    output_coords = create_ssbo(data_template, 5, dtype=np.float32)

    _, _, output_tex = bind_image_texture(target_img_path, 6, GL_WRITE_ONLY)

    # Dispatch shader
    glUseProgram(shader)
    glDispatchCompute((width + 15) // 16, (height + 15) // 16, 1)
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    # Read back indices
    result_indices = copy_shader_buffer(output_indices, np.int32, num_pixels * 4)
    result_indices = result_indices.reshape((height, width, 4))

    # Read back barycentric coordinates
    result_coords = copy_shader_buffer(output_coords, np.float32, num_pixels * 4)
    result_coords = result_coords.reshape((height, width, 4))

    # save texture
    read_and_save_tex(output_tex, "output/mixed.png", width, height)


    glfw.terminate()
    return result_indices, result_coords
