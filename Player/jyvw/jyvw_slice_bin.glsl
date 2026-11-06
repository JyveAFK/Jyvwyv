#[compute]
#version 450

// Pass 1: slice every blob at time t and bin it into the screen tiles it covers.
// One thread per blob. Mirrors stsplat/model.py slice_at() exactly.

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// Blob columns as raw 32-bit words (one column per value; layout in jyvw_file.gd):
// 6 f32 columns, 18 data columns (f16 pairs when compact, else f32), 1 i32 parent column.
layout(set = 0, binding = 0, std430) restrict readonly buffer Blobs { uint d[]; } blobs;
// 4 vec4 per blob: (mean.x, mean.y, conic.a, conic.b), (conic.c, weight, life, id bits),
//                  (rgb, parent group), (velocity px/s, -, -)   [life = (t - t0) / temporal sigma]
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
	uint group_count; // number of group tracks
	uint pad0;
	uint pad1;
	uint pad2;
} pc;

// Group tracks: 9 columns of group_count floats (t0, pos.xy, vel.xy, acc.xy, rot, spin)
layout(set = 0, binding = 5, std430) restrict readonly buffer Groups { float g[]; } groups;

const float CUTOFF = 3.0;
const uint TILE = 16u;
const float GROW_LIMIT = 1.3862944;   // ln(4): matches model.py

// full-record field index -> column (0-5 are f32 columns, 6-23 data columns)
const uint FIELD_COL[24] = uint[24](0u, 1u, 2u, 6u, 4u, 5u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u,
                                    15u, 16u, 17u, 18u, 19u, 20u, 21u, 22u, 23u, 3u);

// words per data column
uint data_stride() {
	return pc.compact != 0u ? (pc.count + (pc.count & 1u)) / 2u : pc.count;
}

// Read field k (full-record numbering) of blob i.
float B(uint i, uint k) {
	uint c = FIELD_COL[k];
	uint n = pc.count;
	if (c < 6u) {
		return uintBitsToFloat(blobs.d[c * n + i]);
	}
	uint base = 6u * n + (c - 6u) * data_stride();
	if (pc.compact == 0u) {
		return uintBitsToFloat(blobs.d[base + i]);
	}
	vec2 pair = unpackHalf2x16(blobs.d[base + i / 2u]);
	return (i & 1u) == 0u ? pair.x : pair.y;
}

int parent_of(uint i) {
	return int(blobs.d[6u * pc.count + 18u * data_stride() + i]);
}

float G(uint gi, uint k) {
	return groups.g[k * pc.group_count + gi];
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
	vec2 vel_n = vec2(B(o, 6u), B(o, 7u)) + vec2(B(o, 8u), B(o, 9u)) * dt;
	float g_angle = 0.0;
	int parent = parent_of(i);
	if (parent >= 0 && uint(parent) < pc.group_count) {
		// follow the group track: centre and angle are relative to it (matches model.slice_at)
		uint gi = uint(parent);
		float dg = pc.t - G(gi, 0u);
		if (pc.period > 0.0) {
			dg -= pc.period * floor(dg / pc.period + 0.5);
		}
		vec2 gpos = vec2(G(gi, 1u), G(gi, 2u)) + vec2(G(gi, 3u), G(gi, 4u)) * dg
		          + 0.5 * vec2(G(gi, 5u), G(gi, 6u)) * dg * dg;
		vec2 gvel = vec2(G(gi, 3u), G(gi, 4u)) + vec2(G(gi, 5u), G(gi, 6u)) * dg;
		g_angle = G(gi, 7u) + G(gi, 8u) * dg;
		float gc = cos(g_angle), gs = sin(g_angle);
		mu = gpos + vec2(gc * mu.x - gs * mu.y, gs * mu.x + gc * mu.y);
		vel_n = gvel + vec2(gc * vel_n.x - gs * vel_n.y, gs * vel_n.x + gc * vel_n.y);
	}
	vec2 grow = clamp(vec2(B(o, 12u), B(o, 13u)) * dt, -GROW_LIMIT, GROW_LIMIT);
	vec2 sc = exp(vec2(B(o, 10u), B(o, 11u)) + grow) * pc.scale_px;
	float th = B(o, 14u) + B(o, 15u) * dt + g_angle;
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
		vec2 tip = m + vel_n * pc.scale_px * pc.arrow_secs;
		lo = min(lo, tip);
		hi = max(hi, tip);
	}
	if (hi.x < 0.0 || hi.y < 0.0 || lo.x >= float(pc.img_w) || lo.y >= float(pc.img_h)) return;

	int tx0 = clamp(int(floor(lo.x / float(TILE))), 0, int(pc.tiles_x) - 1);
	int ty0 = clamp(int(floor(lo.y / float(TILE))), 0, int(pc.tiles_y) - 1);
	int tx1 = clamp(int(floor(hi.x / float(TILE))), 0, int(pc.tiles_x) - 1);
	int ty1 = clamp(int(floor(hi.y / float(TILE))), 0, int(pc.tiles_y) - 1);

	vec2 vel = vel_n * pc.scale_px;
	sliced.s[i * 4u + 0u] = vec4(m, ca, cb);
	sliced.s[i * 4u + 1u] = vec4(cd, w, dt / st, uintBitsToFloat(i));
	sliced.s[i * 4u + 2u] = vec4(col, float(parent));
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
