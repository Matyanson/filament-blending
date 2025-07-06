#version 430

// Output
layout(location = 0) out vec4 fragColor;

// 3D volume of filament IDs (0..255)
layout(binding = 0) uniform usampler3D voxelData;

// Shared SSBOs with one entry per filament ID
layout(std430, binding = 1) buffer BasePoints {
    vec3 base_points[];         // RGB color per filament
};
layout(std430, binding = 2) buffer BasePointsAlpha {
    float base_points_alpha[];  // alpha per single layer
};

// Camera & screen uniforms
uniform ivec2 uScreenSize;       // viewport resolution
uniform float uTime;             // elapsed time in seconds

void main() {
    vec3 volumeSize = vec3(textureSize(voxelData, 0));

    // normalized pixel coords
    vec2 uv = gl_FragCoord.xy / vec2(uScreenSize);
    uv = vec2(uv.x, 1.0 - uv.y);

    float depth = volumeSize.z;           // number of slices
    float speed = 0.25;                     // slices per second
    float z = mod(uTime * speed, 1.0);

    // fetch filament ID
    uint fid = texture(voxelData, vec3(uv, z)).r;

    // look up color (255 means "empty")
    vec3 col = (fid == 255u) 
               ? vec3(0.0) 
               : base_points[fid];

    fragColor = vec4(col, 1.0);
}
