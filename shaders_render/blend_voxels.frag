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

void main() {
    vec3 volumeSize = vec3(textureSize(voxelData, 0));
    
    vec2 uv = gl_FragCoord.xy / vec2(uScreenSize);
    uv = vec2(uv.x, 1.0 - uv.y);
    float depth = volumeSize.z;

    vec3 accum = vec3(0.0);
    for(float i = 0; i < depth; i++) {
        vec3 tc = vec3(uv, i / depth);
        uint fid = texture(voxelData, tc).r;
        if(fid == 255) continue;

        vec3 col = base_points[fid];
        float a1 = base_points_alpha[fid];

        accum = mix(accum, col, a1);
    }

    fragColor = vec4(accum, 1.0);
}
