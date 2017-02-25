#ifndef PPM_H
#define PPM_H

void bundle_sweep2D(GPU_plan*, sdp, body&, sdp FrRot=0.0);

__device__ sdp star_planet_grav_azi_cyl(sdp, sdp, sdp, body&, sdp dt=0.0);

__device__ sdp star_planet_grav_rad_cyl(sdp, sdp, sdp, body&, sdp dt=0.0);

#endif
