#[compute]
#version 450

// Pass 1: slice every blob at time t and bin it into the screen tiles it covers.
// One thread per blob. Mirrors stsplat/model.py slice_at() exactly.

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// Blob records as raw 32-bit words: 24 x f32 (full) or 15 words (compact: 6 x f32 + 18 x f16).
// Layout documented in jyvw_file.gd.
layout(set = 0, binding = 0, std430) restrict readonly buffer Blobs { uint d[]; } blobs;
// 4 vec4 per blob: (mean.x, mean.y, conic.a, conic.b), (conic.c, weight, life, id bits),
//                  (rgb, -), (velocity px/s, -, -)   [life = (t - t0) / temporal sigma]
layout(set = 0, binding = 1, std430) restrict writeonly buffer Sliced { vec4 s[]; } sliced;
layout(set = 0, binding = 2, std430) restrict buffer Counts { uint c[]; } counts;
layout(set = 0, binding = 3, std430) restrict writeonly buffer Lists { uint l[]; } lists;
layout(set = 0, binding = 4, std430) restrict buffer Stats {
	uint live;       // blobs visible this frame
	uint max_tile;   // most blobs requested by a single tile
	uint overflow;   // tile entries dropped because a tile list was full
	uint pad;
} stats;

layout(push_constant, std430) uniform Params {
	float t;          // time in seconds
	float scale_px;   // render width in pixels (normalised units -> pixels)
	uint img_w;
	uint img_h;
	uint tiles_x;
	uint tiles_y;
	uint count;       // number of blobs
	uint cap;         // max entries per tile list
	uint compact;     // 1 = compact records
	uint flags;       // debug flags (bit 2 = motion arrows: widen bins to fit them)
	float arrow_secs; // arrow length in seconds of motion
	float period;     // > 0: looping clip, time wraps with this period
} pc;

const float CUTOFF = 3.0;
const uint TILE = 16u;
const float GROW_LIMIT = 1.3862944;   // ln(4): matches model.py

// full-record field index -> position among the compact record's 16-bit values (-1 = 32-bit)
const int HALF_SLOT[24] = int[24](-1, -1, -1, 0, -1, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                  11, 12, 13, 14, 15, 16, 17, -1);
// full-record field index -> position among the compact record's 32-bit values
const int F32_SLOT[24] = int[24](0, 1, 2, -1, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                 -1, -1, -1, -1, -1, -1, -1, 3);

// Read field k (full-record numbering) of blob i, whichever layout the file uses.
float B(uint i, uint k) {
	if (pc.compact == 0u) {
		return uintBitsToFloat(blobs.d[i * 24u + k]);
	}
	uint base = i * 15u;
	int f = F32_SLOT[k];
	if (f >= 0) {
		return uintBitsToFloat(blobs.d[base + uint(f)]);
	}
	uint h = uint(HALF_SLOT[k]);
	vec2 pair = unpackHalf2x16(blobs.d[base + 6u + h / 2u]);
	return (h & 1u) == 0u ? pair.x : pair.y;
}

void main() {
	uint i = gl_GlobalInvocationID.x;
	if (i >= pc.count) return;
	uint o = i;   // B() takes the blob index

	float dt = pc.t - B(o, 2u);
	if (pc.period > 0.0) {
		// looping clip: use the copy of this blob nearest to t (matches model.wrap_dt)
		dt -= pc.period * floor(dt / pc.period + 0.5);
	} else if (pc.t < B(o, 0u) || pc.t > B(o, 1u)) {
		return;   // hard visibility window (scene cuts / culling)
	}
	float st = exp(B(o, 3u));
	float w = B(o, 22u) * exp(-0.5 * (dt / st) * (dt / st));
	float fade = B(o, 23u);
	if (fade > 0.0 && pc.period <= 0.0) {
		float x = clamp((pc.t - B(o, 0u)) / fade, 0.0, 1.0);
		w *= x * x * (3.0 - 2.0 * x);   // smoothstep, same as model.py
	}
	if (w < 1e-4) return;

	vec2 mu = vec2(B(o, 4u), B(o, 5u)) + vec2(B(o, 6u), B(o, 7u)) * dt
	        + 0.5 * vec2(B(o, 8u), B(o, 9u)) * dt * dt;
	vec2 grow = clamp(vec2(B(o, 12u), B(o, 13u)) * dt, -GROW_LIMIT, GROW_LIMIT);
	vec2 sc = exp(vec2(B(o, 10u), B(o, 11u)) + grow) * pc.scale_px;
	float th = B(o, 14u) + B(o, 15u) * dt;
	vec3 col = 1.0 / (1.0 + exp(-(vec3(B(o, 16u), B(o, 17u), B(o, 18u))
	                             + vec3(B(o, 19u), B(o, 20u), B(o, 21u)) * dt)));

	vec2 m = mu * pc.scale_px;
	float c = cos(th), s = sin(th);
	vec2 sc2 = sc * sc;
	float ix = 1.0 / sc2.x, iy = 1.0 / sc2.y;
	float ca = c * c * ix + s * s * iy;
	float cb = c * s * (ix - iy);
	float cd = s * s * ix + c * c * iy;

	// screen-space extent of the 3-sigma ellipse
	vec2 ext = CUTOFF * sqrt(vec2(c * c * sc2.x + s * s * sc2.y, s * s * sc2.x + c * c * sc2.y));
	vec2 lo = m - ext;
	vec2 hi = m + ext;
	if ((pc.flags & 4u) != 0u) {   // debug arrows extend past the blob: bin them too
		vec2 tip = m + (vec2(B(o, 6u), B(o, 7u)) + vec2(B(o, 8u), B(o, 9u)) * dt) * pc.scale_px * pc.arrow_secs;
		lo = min(lo, tip);
		hi = max(hi, tip);
	}
	if (hi.x < 0.0 || hi.y < 0.0 || lo.x >= float(pc.img_w) || lo.y >= float(pc.img_h)) return;

	int tx0 = clamp(int(floor(lo.x / float(TILE))), 0, int(pc.tiles_x) - 1);
	int ty0 = clamp(int(floor(lo.y / float(TILE))), 0, int(pc.tiles_y) - 1);
	int tx1 = clamp(int(floor(hi.x / float(TILE))), 0, int(pc.tiles_x) - 1);
	int ty1 = clamp(int(floor(hi.y / float(TILE))), 0, int(pc.tiles_y) - 1);

	vec2 vel = (vec2(B(o, 6u), B(o, 7u)) + vec2(B(o, 8u), B(o, 9u)) * dt) * pc.scale_px;
	sliced.s[i * 4u + 0u] = vec4(m, ca, cb);
	sliced.s[i * 4u + 1u] = vec4(cd, w, dt / st, uintBitsToFloat(i));
	sliced.s[i * 4u + 2u] = vec4(col, 0.0);
	sliced.s[i * 4u + 3u] = vec4(vel, 0.0, 0.0);
	atomicAdd(stats.live, 1u);

	for (int ty = ty0; ty <= ty1; ty++) {
		for (int tx = tx0; tx <= tx1; tx++) {
			uint tile = uint(ty) * pc.tiles_x + uint(tx);
			uint k = atomicAdd(counts.c[tile], 1u);
			if (k < pc.cap) {
				lists.l[tile * pc.cap + k] = i;
			} else {
				atomicAdd(stats.overflow, 1u);
			}
			atomicMax(stats.max_tile, k + 1u);
		}
	}
}
