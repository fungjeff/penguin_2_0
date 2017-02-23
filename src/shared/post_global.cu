#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "post_global.h"
#include "cuSafe.cu"

//##################################################################################################

void wait_f_r()
{
  char c;
  cout << " To go on press return! " << endl;
  cin.get(c);
  return;
}

string int_to_string (int num)
{
  stringstream A;
  A << num;
  return (A.str());
}

string frame_num (int num)
{
  string B;
  string C = int_to_string(num);

  if(C.length()==5) B=C;
  else if(C.length()==4) B="0"+C;
  else if(C.length()==3) B="00"+C;
  else if(C.length()==2) B="000"+C;
  else if(C.length()==1) B="0000"+C;
  return (B);
}

string path_to_cwd()
{
  char* a_cwd = getcwd(NULL, 0);
  string s_cwd(a_cwd);
  free(a_cwd);
  return s_cwd;
}

#include "grid.cu"
#include "reduction.cu"
#include "simple_func.cu"


//###################################### GPU CONTROL ############################################
void P2P_all_enable(GPU_plan *set, int nDev)
{
  for (int i=0; i<nDev; i++)
    for (int j=i+1; j<nDev; j++)
    {
      CudaSafeCall( cudaSetDevice(set[i].id) );
      CudaSafeCall( cudaDeviceEnablePeerAccess(set[j].id, 0) );
      CudaSafeCall( cudaSetDevice(set[j].id) );
      CudaSafeCall( cudaDeviceEnablePeerAccess(set[i].id, 0) );
    }
  return;
}

void P2P_all_disable(GPU_plan *set, int nDev)
{
  for (int i=0; i<nDev; i++)
    for (int j=i+1; j<nDev; j++)
    {
      CudaSafeCall( cudaSetDevice(set[i].id) );
      CudaSafeCall( cudaDeviceDisablePeerAccess(set[j].id) );
      CudaSafeCall( cudaSetDevice(set[j].id) );
      CudaSafeCall( cudaDeviceDisablePeerAccess(set[i].id) );
    }

  return;
}

void syncstreams(GPU_plan *set, int nDev)
{
  for (int i=0; i<nDev; i++)
  {
    CudaSafeCall( cudaStreamSynchronize(set[i].stream) );
  }
  return;
}

void syncdevices(GPU_plan *set, int nDev)
{
  for (int i=0; i<nDev; i++)
  {
    CudaSafeCall( cudaSetDevice(set[i].id) );
    CudaSafeCall( cudaDeviceSynchronize() );
  }
  return;
}

void syncallevents(GPU_plan *set, int nDev)
{
  for (int i=0; i<nDev; i++)
  {
    CudaSafeCall( cudaEventSynchronize( set[i].event ) );
  }
  return;
}

void syncallevents(int n, GPU_plan *set, int nDev)
{
  for (int i=0; i<nDev; i++)
  {
    CudaSafeCall( cudaStreamWaitEvent( set[n].stream, set[i].event ,0) );
  }
  return;
}

void reset_devices(int nDev)
{
  int d;
  cudaGetDevice(&d);

  printf (" Current Device : %i\n",d);
  for (int n=0; n<nDev; n++)
  {
    CudaSafeCall( cudaSetDevice(n) );
    CudaSafeCall( cudaDeviceSynchronize() );
    CudaSafeCall( cudaDeviceReset() );
    printf(" Device %i terminated.",n);
  }
  exit(-1);
}
