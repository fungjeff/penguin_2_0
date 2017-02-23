#ifndef PPM_H
#define PPM_H

void bundle_sweep2D(GPU_plan*, sdp, body&);
//void bundle_sweep3D(GPU_plan*, sdp, body&);

__device__ sdp star_planet_grav_azi_cyl(sdp, sdp, sdp, body&, sdp dt=0.0);

#endif
