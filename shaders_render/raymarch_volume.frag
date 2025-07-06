#version 430

// Output
layout(location = 0) out vec4 fragColor;

// 3D volume of filament IDs (0..255)
layout(binding = 0) uniform usampler3D voxelData;

// Camera & screen uniforms
uniform vec3 uCameraPos;         // world-space camera position
uniform ivec2 uScreenSize;       // viewport resolution
uniform float uStepSize;         // step length in world units

vec2 intersectBox(vec3 ro, vec3 rd, vec3 boxMin, vec3 boxMax) {
    vec3 inv = 1.0 / rd;
    vec3 t0  = (boxMin - ro) * inv;
    vec3 t1  = (boxMax - ro) * inv;
    vec3 tmin = min(t0, t1);
    vec3 tmax = max(t0, t1);
    float tn = max(max(tmin.x, tmin.y), tmin.z);
    float tf = min(min(tmax.x, tmax.y), tmax.z);
    return vec2(tn, tf);
}

void main() {
    fragColor = vec4(0.0);
    vec3 uVolumeOrigin = vec3(0.0);
    vec3 volumeSize = vec3(textureSize(voxelData, 0));

    // 1) map ndc
    vec2 uv = gl_FragCoord.xy / vec2(uScreenSize);
    vec2 ndc = uv * 2.0 - 1.0;

    // vec3 rayOrigin = uCameraPos;
    // vec3 rayDir = normalize(vec3(ndc, -1.0));
    
    vec3 rayOrigin = uCameraPos;
    vec3 rayTarget = vec3(volumeSize / 2.0);
    vec3 forward = normalize(rayTarget - rayOrigin);
    vec3 right   = normalize(cross(forward, vec3(0.0, 1.0, 0.0)));
    vec3 up      = cross(right, forward);

    vec3 rayDir = normalize(
        ndc.x * right +
        ndc.y * up +
        1.0 * forward
    );

    // 2) test intersection against our cuboid
    vec3 boxMin = uVolumeOrigin;
    vec3 boxMax = uVolumeOrigin + volumeSize;
    vec2 tInterval = intersectBox(rayOrigin, rayDir, boxMin, boxMax);
    float tn = tInterval.x;
    float tf = tInterval.y;
    bool hit = (tn <= tf) && (tf >= 0.0);
    if(!hit) return;

    // 3) itterate from tn to tf by step-size
    float accum = 0.0;
    float volumeAlpha = 0.03;
    float t = tn;
    
    for (int i = 0; i < 512; ++i) {
        accum += uStepSize * volumeAlpha * (1.0 - accum);
        t += uStepSize;

        if(accum > 0.995 || t > tf)
            break;
    }

    fragColor = vec4(accum, 0.0, accum, 1.0);
}