import time
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram
import glfw


def eye_position_from(view_mat: np.ndarray) -> np.ndarray:
    """
    Given a 4×4 view matrix (camera transform world→view),
    returns the world‐space camera position (the “eye”).
    This is simply the inverse‐transform of the origin.

    :param view_mat: 4×4 numpy array
    :return: 3‐element numpy array camera position
    """
    # Invert the view matrix to get view→world
    inv = np.linalg.inv(view_mat)
    # The world‐space position of the camera is where the view‐space origin maps to:
    # inv @ [0,0,0,1]^T
    eye_homog = inv @ np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return eye_homog[:3] / eye_homog[3]

class VolumeRenderer:
    def __init__(self, window, w_width, w_height, vert_path, frag_path):
        # 2) Set window hints for a core profile
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        # 5) Store screen info
        self.window = window
        self.w_width  = w_width
        self.w_height = w_height

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
        self.uInvViewProj_loc = glGetUniformLocation(self.prog, 'uInvViewProj')
        self.uCameraPos_loc   = glGetUniformLocation(self.prog, 'uCameraPos')
        self.uScreenSize_loc  = glGetUniformLocation(self.prog, 'uScreenSize')
        self.uVolumeSize_loc  = glGetUniformLocation(self.prog, 'uVolumeSize')
        self.uStepSize_loc    = glGetUniformLocation(self.prog, 'uStepSize')
        self.uTime_loc        = glGetUniformLocation(self.prog, 'uTime')


    def set_volume(self, voxel_tex, volume_size):
        """voxel_tex: GL texture ID bound at unit 0; volume_size: (W,H,D) in world units"""
        self.voxel_tex   = voxel_tex
        self.volume_size = volume_size
        # Texture unit 0 → sampler binding 0 in GLSL
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_3D, self.voxel_tex)
        # Tell your shader to sample from unit 0
        glUniform1i(glGetUniformLocation(self.prog, "voxelData"), 0)


    def set_camera(self, view_mat, proj_mat):
        self.invVP      = np.linalg.inv(proj_mat @ view_mat).astype(np.float32)
        self.camera_pos = eye_position_from(view_mat).astype(np.float32)

    def render_frame(self, step_size=1.0):
        # Bind shader & VAO
        glUseProgram(self.prog)
        glBindVertexArray(self.quad_vao)

        # Upload uniforms
        glUniformMatrix4fv(self.uInvViewProj_loc, 1, GL_FALSE, self.invVP)
        glUniform3f(self.uCameraPos_loc,   *self.camera_pos)
        glUniform2i(self.uScreenSize_loc,  self.w_width, self.w_height)
        glUniform3f(self.uVolumeSize_loc,  *self.volume_size)
        glUniform1f(self.uStepSize_loc,    step_size)
        glUniform1f(self.uTime_loc,    glfw.get_time())

        # Draw quad (uses gl_VertexID)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

        # Swap and poll events
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def cleanup(self):
        glfw.terminate()
