#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "variable_types.h"
#include "global.h"
#include "cuSafe.cu"

//#########################################################################################

__global__ void dt_lv1(hydr_ring *rings, sdp *dt_2D)
{
  int i = blockIdx.x;
  int k = blockIdx.y;
  int j;
  int ik = i + gridDim.x * k;

  int tt = threadIdx.x;
  int tmax = blockDim.x;
  int loop = jmax/tmax;
  if (jmax%tmax) loop++;

  __shared__ sdp ridt[1024];

  sdp tmp[ndim];
  sdp val = 0.0;
  sdp cs, widthx, widthy, widthz, rad, rad_cyl;

  for (int l=0; l<loop; l++)
  {
    j = tt + tmax * l;

    if (j<jmax)
    {
      widthz = rings[ik].dz;
      widthy = rings[ik].dy[j];
      widthx = rings[ik].dx;
      rad = rings[ik].xc;
      rad_cyl = rad;

      if (ngeomz==5) rad_cyl *= csin(rings[ik].zc);

      if (ngeomy> 2) widthy *= rad_cyl;

      if (ngeomz> 2) widthz *= rad;

      cs = csqrt(gam*rings[ik].p[j]/rings[ik].r[j]);

      tmp[0] = (cabs(rings[ik].u[j]) + cs) / widthx;
      tmp[1] = (cabs(rings[ik].v[j] - rings[ik].rot_v - FrRot*rad_cyl) + cs) / widthy;
      #if ndim==3
      tmp[2] = (cabs(rings[ik].w[j]) + cs) / widthz;
      ridt[tt] = max3(tmp[0],tmp[1],tmp[2]);
      #else
      ridt[tt] = cmax(tmp[0],tmp[1]);
      #endif
    }
    else
    {
      ridt[tt] = 0.0;  
    }
    __syncthreads();

    bin_reduc_max(1024, ridt);

    if (tt==0) if (ridt[0]>val) val=ridt[0];
    __syncthreads();
  }
  if (tt==0) dt_2D[ik]=val;

  return;
}


__global__ void dt_lv2(sdp *dt_2D, sdp *dt_1D)
{
  int i = blockIdx.x;
  int k = threadIdx.x;
  int ik = i + gridDim.x * k;

  __shared__ sdp ridt[kmax];

  ridt[k] = dt_2D[ik];
  __syncthreads();

  round_reduc_max(kmax, ridt);

  if (k==0) dt_1D[i]=ridt[0];
  
  return;
}

__global__ void dt_lv3(sdp *dt_1D, sdp *output, int iblk)
{
  int i = threadIdx.x;

  __shared__ sdp ridt[1024];
  if (i<iblk)  ridt[i] = dt_1D[i];
  else         ridt[i] = 0.0;
  __syncthreads();

  bin_reduc_max(1024, ridt);

  if (i==0) *output =  courant / ridt[0];

  return;
}

//=========================================================================================

sdp get_dt(sdp dt, GPU_plan *set)
{
  for (int n=0; n<nDev; n++)
  {
    CudaSafeCall( cudaSetDevice(set[n].id) );

    dt_lv1<<< set[n].t_grid , 1024 , 0 , set[n].stream >>>(set[n].rings, set[n].dt_2D);
    CudaCheckError();
    dt_lv2<<< set[n].iblk , set[n].kblk , 0 , set[n].stream >>>(set[n].dt_2D, set[n].dt_1D);
    CudaCheckError();
    dt_lv3<<< 1 , 1024 , 0 , set[n].stream >>>(set[n].dt_1D, set[n].dt, set[n].iblk);
    CudaCheckError();

    CudaSafeCall( cudaMemcpyAsync( set[n].h_dt, set[n].dt, sizeof(sdp), cudaMemcpyDeviceToHost, set[n].stream) );
  }

  for (int n=0; n<nDev; n++) CudaSafeCall( cudaStreamSynchronize(set[n].stream) );

  dt = 1.1*dt; // limiting constraint on rate of increase of dt
  for (int n=0; n<nDev; n++)
  {
    //cout << "host:" << dt << endl;
    //cout << "device:" << *set[n].h_dt << endl;
    dt= min(dt, *set[n].h_dt);
  }

  if (dt/endtime < 1.0e-12)
  {
    printf( " time step too small: %e\n",dt/endtime);
  }

  return dt;
}
