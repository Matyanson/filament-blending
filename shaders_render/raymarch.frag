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
uniform vec3 uCameraPos;         // world-space camera position
uniform ivec2 uScreenSize;       // viewport resolution
uniform float uStepSize;         // step length in world units

// returns (tNear, tFar)
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
    vec3 volumeSize = vec3(textureSize(voxelData, 0));

    // 1) Generate a ray in world-space through this pixel
    vec2 uv = gl_FragCoord.xy / vec2(uScreenSize);
    uv = vec2(uv.x, 1.0 - uv.y);
    vec2 ndc = uv * 2.0 - 1.0;
    
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


    // 2) get entry/exit
    vec3 boxMin = vec3(0.0);
    vec3 boxMax = volumeSize;
    vec2 ts = intersectBox(rayOrigin, rayDir, boxMin, boxMax);
    if (ts.y <= max(ts.x, 0.0)) {
        // no intersection
        fragColor = vec4(0.0);
        return;
    }


    // 3) March from the camera into the volume
    // start at entry
    float t = max(ts.x, 0.0);
    vec3 pos = rayOrigin + rayDir * (t + 0.001);
    vec4 accum = vec4(0.0);  // RGBA accumulation

    for (int i = 0; i < 100000; i++) {
        // 3a) Compute normalized texture coords [0,1]
        vec3 tc = pos / volumeSize;
        if (any(lessThan(tc, vec3(0.0))) || any(greaterThan(tc, vec3(1.0)))) {
            break;
        }

        // 3b) Fetch filament ID (integer) from the 3D volume
        uint fid = texture(voxelData, tc).r;
        if(fid == 255) {
            // 3f) Advance along the ray
            pos += rayDir * uStepSize;
            continue;
        }

        // 3c) Lookup base color and single-layer opacity
        vec3 col = base_points[fid];
        float a1 = base_points_alpha[fid];  // alpha of material per 1px
        a1 = a1 * uStepSize;                // alpha of material per 1 step

        // 3d) Compute this sample’s contribution
        //    α_sample = 1–(1–a1) layering for one voxel
        //    then modulate by remaining transparency
        float alpha = a1 * (1.0 - accum.a);
        accum.rgb  += alpha * col;
        accum.a    += alpha;

        // 3e) Early exit if opaque enough
        if (accum.a > 0.995) {
            break;
        }

        // 3f) Advance along the ray
        pos += rayDir * uStepSize;
    }

    // 4) Output
    fragColor = accum;
}
