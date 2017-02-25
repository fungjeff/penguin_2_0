#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "variable_types.h"
#include "global.h"
#include "disk_profile.h"
#include "cuSafe.cu"

//================================================================================
__global__ void Index_Shift(hydr_ring *rings, SymDisk *val, sdp iblk, sdp dt, sdp FrRot)
{
  int n = blockIdx.x + gridDim.x * blockIdx.y;
  sdp rot_v, jp;
  sdp widthy;

  widthy = rings[n].xc*rings[n].dy[0] / dt;

  #if FARGO_flag == 1
  int idx = rings[n].i + iblk * rings[n].k;
  rot_v = val[idx].v - FrRot*rings[n].xc;
  #elif FARGO_flag == 2
  rot_v = (1.0 - FrRot)*rings[n].xc;
  #else
  rot_v = 0.0;
  #endif

  jp = (rings[n].rot_v/widthy);
  if (jp>=0.0) 
  {
    rings[n].inc_j = (int)(jp+0.5);
    rings[n].res_v = floor(jp+0.5)*widthy;
  }
  else
  {
    rings[n].inc_j = (int)(jp-0.5);
    rings[n].res_v = floor(jp+0.5)*widthy;
  }
  rings[n].rot_v = rot_v;

  return;
}

void Index_Shift(GPU_plan *set, sdp dt, sdp FrRot)
{
  for(int n=0; n<nDev; n++)
  {
    CudaSafeCall( cudaSetDevice(set[n].id) );

    Index_Shift<<< set[n].sy_grid , 1 , 0 , set[n].stream >>> (set[n].rings, set[n].val, set[n].iblk, dt, FrRot);
    CudaCheckError();
  }
/*
  for (int n=0; n<nDev; n++)
  {
    CudaSafeCall( cudaMemcpy( set[n].h_rings, set[n].rings, set[n].memsize, cudaMemcpyDeviceToHost ) );
    for (int i=0; i<set[n].iblk; i++)
    {
      printf(" %f %f %f %i %i \n", set[n].h_rings[i].xc, set[n].h_rings[i].rot_v*dt/set[n].h_rings[i].dy[0]/set[n].h_rings[i].xc, (set[n].h_rings[i].res_v)*dt/set[n].h_rings[i].dy[0]/set[n].h_rings[i].xc, set[n].h_rings[i].inc_j, set[n].h_rings[i].rot_j);
    }
  }
  cout << FrRot << endl;
  wait_f_r();
*/
  syncstreams(set);

  return;
}

bool Check_Shift(hydr_ring *rings)
{
  bool shear = false;
  int n;

  for (int i=1; i<imax-1; i++)
  {
    for (int k=1; k<kmax-1; k++)
    {
      n = i+imax*k;
      if (abs(rings[n].rot_j-rings[n+1].rot_j) > 1) shear = true;
      if (abs(rings[n].rot_j-rings[n-1].rot_j) > 1) shear = true;
      if (abs(rings[n].rot_j-rings[n+imax].rot_j) > 1) shear = true;
      if (abs(rings[n].rot_j-rings[n-imax].rot_j) > 1) shear = true;
    }
  }

  return shear;
}
