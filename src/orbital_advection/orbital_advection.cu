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
__global__ void Index_Shift(hydr_ring *rings, sdp dt)
{
  int n = blockIdx.x + gridDim.x * blockIdx.y;
  sdp jp;
  sdp widthy;

  widthy = rings[n].xc*rings[n].dy[0] / dt;
  jp = (rings[n].rot_v/widthy);
  if (jp>=0.0) 
  {
    rings[n].inc_j = (int)(jp+0.5);
    rings[n].res_v = rings[n].rot_v - (floor(jp+0.5)*widthy);
  }
  else
  {
    rings[n].inc_j = (int)(jp-0.5);
    rings[n].res_v = rings[n].rot_v - (floor(jp+0.5)*widthy);
  }

  //if (n==0) printf("%i : %f %i %f %f\n", n, dt, rings[n].inc_j, rings[n].res_v, rings[n].rot_v);
  //if (n==0) printf("%f %i %i %f\n", jp, rings[n].rot_j, rings[n].inc_j, rings[n].rot_v*cpow(rings[n].xc,0.5));

  return;
}

void Index_Shift(GPU_plan *set, sdp dt)
{
  for(int n=0; n<nDev; n++)
  {
    CudaSafeCall( cudaSetDevice(set[n].id) );

    Index_Shift<<< set[n].sy_grid , 1 , 0 , set[n].stream >>> (set[n].rings, dt);
    CudaCheckError();
  }

  for (int i=0; i<nDev; i++)  CudaSafeCall( cudaStreamSynchronize(set[i].stream) );
/*
  for(int i=0; i<imax; i++)
  {
    CudaSafeCall( cudaMemcpy( set[0].h_rings, set[0].rings, set[0].memsize, cudaMemcpyDeviceToHost ) );
    printf("%i : %f %i %f(%f) %f\n", i, dt, set[0].h_rings[i].inc_j, set[0].h_rings[i].res_v, set[0].h_rings[i].res_v*dt/set[0].h_rings[i].dy[0]/set[0].h_rings[i].xc, set[0].h_rings[i].rot_v);
    printf("%i : %e\n", i, set[0].h_rings[i].v[0]*pow(set[0].h_rings[i].xc,0.5)-1.0);
  }
    wait_f_r();
*/
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
