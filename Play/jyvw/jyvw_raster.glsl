#[compute]
#version 450

// Pass 2: one workgroup per 16x16 tile. Every pixel sums the blobs binned to its
// tile and outputs sum(w * rgb) / sum(w) (normalised blending, no sorting).
// Optional debug views and overlays are selected with `mode` and `flags`.

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) restrict readonly buffer Sliced { vec4 s[]; } sliced;
layout(set = 0, binding = 1, std430) restrict readonly buffer Counts { uint c[]; } counts;
layout(set = 0, binding = 2, std430) restrict readonly buffer Lists { uint l[]; } lists;
layout(rgba8, set = 0, binding = 3) uniform restrict writeonly image2D out_img;
layout(set = 0, binding = 4, std430) restrict writeonly buffer Pick { uint p[]; } pick;

layout(push_constant, std430) uniform Params {
	uint img_w;
	uint img_h;
	uint tiles_x;
	uint cap;
	vec4 background;   // shown where no blob has any weight
	uint mode;         // 0 image, 1 coverage, 2 dominant blob, 3 life stage
	uint flags;        // 1 outlines, 2 centres, 4 arrows, 8 write pick buffer,
	                   // 16 outline in blob colour, 32 outline in unique colour
	uint selected;     // highlighted blob id (0xFFFFFFFF = none)
	float ring_k;      // outline radius in sigmas
	float arrow_secs;  // motion arrow length in seconds
	float pad0;
	float pad1;
	float pad2;
} pc;

const float CUTOFF2 = 9.0;   // 3 sigma squared
const uint BATCH = 256u;
const uint NONE = 0xFFFFFFFFu;

shared vec4 sh0[BATCH];
shared vec4 sh1[BATCH];
shared vec4 sh2[BATCH];
shared vec4 sh3[BATCH];

vec3 id_colour(uint id) {
	uint h = id * 2654435761u;
	h ^= h >> 15;
	h *= 2246822519u;
	h ^= h >> 13;
	return vec3(float(h & 255u), float((h >> 8) & 255u), float((h >> 16) & 255u)) / 255.0 * 0.75 + 0.25;
}

// black -> blue -> green -> yellow -> white
vec3 heat(float x) {
	x = clamp(x, 0.0, 1.0);
	vec3 c0 = vec3(0.02, 0.02, 0.08);
	vec3 c1 = vec3(0.10, 0.25, 0.95);
	vec3 c2 = vec3(0.10, 0.80, 0.30);
	vec3 c3 = vec3(1.00, 0.85, 0.10);
	vec3 c4 = vec3(1.00, 1.00, 1.00);
	if (x < 0.25) return mix(c0, c1, x / 0.25);
	if (x < 0.50) return mix(c1, c2, (x - 0.25) / 0.25);
	if (x < 0.75) return mix(c2, c3, (x - 0.50) / 0.25);
	return mix(c3, c4, (x - 0.75) / 0.25);
}

// fading in (before peak) = green, peak = white, fading out = red
vec3 life_colour(float x) {
	float k = clamp(abs(x) / 3.0, 0.0, 1.0);
	return x < 0.0 ? mix(vec3(1.0), vec3(0.1, 0.9, 0.2), k) : mix(vec3(1.0), vec3(1.0, 0.15, 0.1), k);
}

void main() {
	uint tile = gl_WorkGroupID.y * pc.tiles_x + gl_WorkGroupID.x;
	uint n = min(counts.c[tile], pc.cap);
	uint lid = gl_LocalInvocationIndex;
	uvec2 p = gl_GlobalInvocationID.xy;
	vec2 pos = vec2(p) + 0.5;
	bool debug = pc.mode != 0u || (pc.flags & 15u) != 0u || pc.selected != NONE;
	float ring_lim = (pc.ring_k + 0.6) * (pc.ring_k + 0.6);

	vec4 acc = vec4(0.0);
	vec3 life_acc = vec3(0.0);
	float best_g = 0.0;
	uint best_id = NONE;
	float ring_a = 0.0;
	vec3 ring_c = vec3(1.0);
	float sel_a = 0.0;
	float dot_a = 0.0;
	float arrow_a = 0.0;

	for (uint base = 0u; base < n; base += BATCH) {
		uint idx = base + lid;
		if (idx < n) {
			uint j = lists.l[tile * pc.cap + idx];
			sh0[lid] = sliced.s[j * 4u + 0u];
			sh1[lid] = sliced.s[j * 4u + 1u];
			sh2[lid] = sliced.s[j * 4u + 2u];
			sh3[lid] = sliced.s[j * 4u + 3u];
		}
		barrier();
		uint m = min(BATCH, n - base);
		for (uint k = 0u; k < m; k++) {
			vec4 a = sh0[k];
			vec4 b = sh1[k];
			vec2 d = pos - a.xy;
			float q = a.z * d.x * d.x + 2.0 * a.w * d.x * d.y + b.x * d.y * d.y;
			if (q < CUTOFF2) {
				float g = b.y * exp(-0.5 * q);
				acc += vec4(sh2[k].rgb * g, g);
				if (debug) {
					life_acc += life_colour(b.z) * g;
					if (g > best_g) {
						best_g = g;
						best_id = floatBitsToUint(b.w);
					}
				}
			}
			if (!debug) continue;
			uint id = floatBitsToUint(b.w);
			float vis = clamp(b.y * 3.0, 0.2, 1.0);   // overlays fade with the blob's weight
			bool is_sel = id == pc.selected;
			if (((pc.flags & 1u) != 0u || is_sel) && q < ring_lim && q > 1e-8) {
				// distance in pixels from the k-sigma ellipse
				float r = sqrt(q);
				vec2 grad = vec2(a.z * d.x + a.w * d.y, a.w * d.x + b.x * d.y);
				float px = abs(r - pc.ring_k) * r / max(length(grad), 1e-6);
				if (is_sel) {
					sel_a = max(sel_a, clamp(2.0 - px, 0.0, 1.0));
				} else {
					float ra = clamp(1.0 - px, 0.0, 1.0) * vis;
					if (ra > ring_a) {
						ring_a = ra;
						ring_c = (pc.flags & 32u) != 0u ? id_colour(id)
						       : (pc.flags & 16u) != 0u ? sh2[k].rgb : vec3(1.0);
					}
				}
			}
			if ((pc.flags & 2u) != 0u || is_sel) {
				dot_a = max(dot_a, clamp(2.2 - length(d), 0.0, 1.0) * (is_sel ? 1.0 : vis));
			}
			if ((pc.flags & 4u) != 0u) {
				vec2 v = sh3[k].xy * pc.arrow_secs;
				float len2 = dot(v, v);
				if (len2 > 4.0) {
					float h = clamp(dot(d, v) / len2, 0.0, 1.0);
					arrow_a = max(arrow_a, clamp(1.25 - length(d - v * h), 0.0, 1.0) * vis);
				}
			}
		}
		barrier();
	}

	if (p.x >= pc.img_w || p.y >= pc.img_h) return;
	float cover = clamp(acc.a * 1e4, 0.0, 1.0);
	vec3 img = mix(pc.background.rgb, acc.rgb / (acc.a + 1e-6), cover);
	vec3 rgb = img;
	if (pc.mode == 1u) {
		// log10 total weight: 1e-4 (barely covered) .. 10 (heavily covered)
		rgb = heat((log(acc.a + 1e-7) / log(10.0) + 4.0) / 5.0);
	} else if (pc.mode == 2u) {
		float lum = dot(img, vec3(0.299, 0.587, 0.114));
		rgb = best_id == NONE ? pc.background.rgb : id_colour(best_id) * (0.35 + 0.65 * lum);
	} else if (pc.mode == 3u) {
		float lum = dot(img, vec3(0.299, 0.587, 0.114));
		rgb = mix(pc.background.rgb, life_acc / (acc.a + 1e-6), cover) * (0.3 + 0.7 * lum);
	}
	if (debug) {
		rgb = mix(rgb, ring_c, ring_a * 0.85);
		rgb = mix(rgb, vec3(1.0, 0.8, 0.15), arrow_a);
		rgb = mix(rgb, vec3(0.2, 1.0, 1.0), sel_a);
		rgb = mix(rgb, vec3(1.0), dot_a);
	}
	if ((pc.flags & 8u) != 0u) {
		pick.p[p.y * pc.img_w + p.x] = best_id;
	}
	imageStore(out_img, ivec2(p), vec4(clamp(rgb, 0.0, 1.0), 1.0));
}
