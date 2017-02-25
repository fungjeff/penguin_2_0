#ifndef PLANET_H
#define PLANET_H

__host__ __device__ sdp get_L(body&, sdp);
__host__ __device__ sdp get_E(body&, sdp);
__host__ __device__ sdp get_ecc(body&, sdp);
void init_planet(body&, sdp&);
void planet_forces(body&, sdp, GPU_plan*, sdp);
void kick_drift_kick(body&, sdp&, GPU_plan*, sdp, sdp, bool);

#endif
