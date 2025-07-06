import numpy as np


def get_voxel_volume(filament_order, layers: np.ndarray):
    # 1) Build the voxel grid: uint8 IDs in shape (W, H, D)
    #    Depth is the max over all pixel-layer sums
    width = layers.shape[1]
    height = layers.shape[0]
    depth = int(np.max(np.sum(layers, axis=2)))
    volume_dimensions = (width, height, depth)
    voxel_data = np.full((height, width, depth), 255, dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            z = 0
            for i, thickness in enumerate(layers[y, x]):
                if thickness <= 0:
                    continue
                
                fil_id = filament_order[i]
                # fill `thickness` voxels with `fil_id`
                end = min(depth, z + thickness)
                voxel_data[y, x, z:end] = fil_id
                z = end
    
    # 2) Convert to (depth, height, width) using np.transpose
    voxel_data = np.transpose(voxel_data, (2, 0, 1))

    return voxel_data, volume_dimensions