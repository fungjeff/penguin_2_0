#ifndef DISK_PROFILE_H
#define DISK_PROFILE_H

//=======================================================================
// 2D disk
//=======================================================================

sdp get_rho(sdp);

__host__ __device__ sdp get_cs2(sdp);
__host__ __device__ sdp get_h(sdp);

sdp get_P(sdp);

sdp get_dP_dr(sdp);

__host__ __device__ sdp get_nu(sdp);

sdp get_viscous_vr(sdp);

//=======================================================================
// 3D disk
//=======================================================================

sdp get_rho(sdp, sdp);

sdp get_P(sdp, sdp);

sdp get_dP_dr(sdp r, sdp z);

sdp get_dP_s(sdp r, sdp z);

sdp get_viscous_vr(sdp, sdp);

//=======================================================================
// Other
//=======================================================================

sdp set_M_p(sdp);

#endif
