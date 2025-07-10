import time
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram
import glfw

from main_dispatcher import get_glsl_format


def set_3d_texture_as_sampler(data_slices):
    dtype = data_slices.dtype
    internal_format, pixel_format, pixel_type = get_glsl_format(dtype)

    d, h, w = data_slices.shape
    data_flat = data_slices.flatten()
    tex = glGenTextures(1)
    tex = int(tex)
    glBindTexture(GL_TEXTURE_3D, tex)

    # No alignment padding
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

    # Allocate storage
    glTexStorage3D(GL_TEXTURE_3D, 1, internal_format, w, h, d)
    # Upload your uint8 ID data
    glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, w, h, d,
                    pixel_format, pixel_type, data_flat)

    # Nearest‐neighbor, no interpolation
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

    return tex

class VolumeRenderer:
    def __init__(self, window, w_width, w_height, vert_path, frag_path):
        # 2) Set window hints for a core profile
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        # 5) Store screen & control info
        self.window = window
        self.w_width  = w_width
        self.w_height = w_height

        self.camera_pos = np.array([0, 0, 100.0], dtype=np.float32)
        self.camera_speed = 1.0

        # Set controller key callback
        glfw.set_key_callback(window, self.key_callback)

        # 6) Build fullscreen-quad shader
        vert_src = open(vert_path).read()
        frag_src = open(frag_path).read()
        vs = compileShader(vert_src, GL_VERTEX_SHADER)
        fs = compileShader(frag_src, GL_FRAGMENT_SHADER)
        self.prog = compileProgram(vs, fs)
        glUseProgram(self.prog) 

        # 7) Create a VAO for the quad (no VBO needed if using gl_VertexID)
        self.quad_vao = glGenVertexArrays(1)
        glBindVertexArray(self.quad_vao)

        # 8) Pre-cache uniform locations
        self.uCameraPos_loc   = glGetUniformLocation(self.prog, 'uCameraPos')
        self.uScreenSize_loc  = glGetUniformLocation(self.prog, 'uScreenSize')
        self.uStepSize_loc    = glGetUniformLocation(self.prog, 'uStepSize')
        self.uTime_loc        = glGetUniformLocation(self.prog, 'uTime')

        # 9) set uniforms
        glUniform2i(self.uScreenSize_loc,  self.w_width, self.w_height)

    def key_callback(self, window, key, scancode, action, mods):
        if(action == glfw.REPEAT):
            self.camera_speed = min(self.camera_speed + 0.1, 100.0)
        else: self.camera_speed = 1.0

        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_UP:
                self.camera_pos[1] -= self.camera_speed  # move up
            elif key == glfw.KEY_DOWN:
                self.camera_pos[1] += self.camera_speed  # move down
            elif key == glfw.KEY_LEFT:
                self.camera_pos[0] -= self.camera_speed  # move left
            elif key == glfw.KEY_RIGHT:
                self.camera_pos[0] += self.camera_speed  # move right
            elif key == glfw.KEY_W:
                self.camera_pos[2] -= self.camera_speed  # move forward
            elif key == glfw.KEY_S:
                self.camera_pos[2] += self.camera_speed  # move backward


    def set_volume(self, unit, voxel_data):
        """voxel_tex: GL texture ID bound at unit 0; volume_size: (W,H,D) in world units"""

        self.voxel_tex = set_3d_texture_as_sampler(voxel_data)
        # Texture unit 0 → sampler binding 0 in GLSL
        glActiveTexture(GL_TEXTURE0 + unit)
        glBindTexture(GL_TEXTURE_3D, self.voxel_tex)
        # Tell your shader to sample from unit 0
        glUniform1i(glGetUniformLocation(self.prog, "voxelData"), 0)


    def set_camera(self, camera_pos):
        self.camera_pos = camera_pos

    def render_frame(self, step_size=1.0):
        # Bind shader & VAO
        glUseProgram(self.prog)
        glBindVertexArray(self.quad_vao)

        # Upload uniforms
        glUniform3f(self.uCameraPos_loc,   *self.camera_pos)
        glUniform1f(self.uStepSize_loc,    step_size)
        glUniform1f(self.uTime_loc,    glfw.get_time())

        # Draw quad (uses gl_VertexID)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

        # Swap and poll events
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def cleanup(self):
        glfw.terminate()
