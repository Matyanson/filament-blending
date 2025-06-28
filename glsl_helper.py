import numpy as np

def look_at(eye, center, up):
    """
    Build a right-handed look-at view matrix.
    
    :param eye:    (3,) camera position
    :param center: (3,) look-at target
    :param up:     (3,) up direction (usually [0,1,0])
    :return:       (4,4) view matrix
    """
    f = center - eye
    f = f / np.linalg.norm(f)
    # pokud by f a up byly rovnoběžné, up' = něco jiného
    if abs(np.dot(f, up)) > 0.999:
        up = np.array([0, 1, 0], dtype=np.float32)
    u = up / np.linalg.norm(up)
    s = np.cross(f, u)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    
    M = np.eye(4, dtype=np.float32)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = -eye
    return M @ T

def perspective(fovy, aspect, znear, zfar):
    """
    Build a right-handed perspective projection matrix (OpenGL style).
    
    :param fovy:   vertical field of view in radians
    :param aspect: width/height
    :param znear:  near clip (positive)
    :param zfar:   far clip  (positive)
    :return:       (4,4) proj matrix
    """
    f = 1.0 / np.tan(fovy * 0.5)
    M = np.zeros((4,4), dtype=np.float32)
    M[0,0] = f / aspect
    M[1,1] = f
    M[2,2] = (zfar + znear) / (znear - zfar)
    M[2,3] = (2 * zfar * znear) / (znear - zfar)
    M[3,2] = -1.0
    return M
