#version 430

out vec2 vUv;

void main() {
    // +1/−1 quad positions and UVs (0–1)
    const vec2 pos[4] = vec2[](
        vec2(-1, -1), vec2(+1, -1),
        vec2(-1, +1), vec2(+1, +1)
    );
    const vec2 uv[4] = vec2[](
        vec2(0, 0), vec2(1, 0),
        vec2(0, 1), vec2(1, 1)
    );
    int idx = gl_VertexID;
    gl_Position = vec4(pos[idx], 0.0, 1.0);
    vUv = uv[idx];
}
