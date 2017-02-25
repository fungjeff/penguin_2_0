#ifndef GLOBAL_H
#define GLOBAL_H

typedef double sdp;
typedef double sdp;
typedef double2 sdp2;

using namespace std;

//=======================================================================
// GPU Arch and Grid Dim
//=======================================================================

#define ndim 2

const int n_pad = 6;     //number of boundary zones on each side: 6 for ppm

const int arrsize = 96;  //for optimal speed: arrsize = 32*(integer)
const int realarr = arrsize - 2*n_pad;

const int imax = 850;    //for optimal speed: imax = realarr*(integer)
const int jmax = 2000;    //                  same for jmax and kmax
const int kmax = 1;

//=======================================================================
// Constants
//=======================================================================

const sdp hpi = 1.570796326794896619231321691639751442;
const sdp pi = 3.1415926535897932384626433832795;
const sdp twopi = 6.283185307179586476925286766559;
const sdp third = 0.3333333333333333333333333;
const sdp fourthd = 1.3333333333333333333333333;
const sdp EarthMass = 0.000003;
const sdp NeptuneMass = 0.00005;
const sdp JupiterMass = 0.001;
const sdp smallp = 1.0e-15;
const sdp smallr = 1.0e-15;

//=======================================================================
// Planet parameters
//=======================================================================

const sdp M_p = 5.0*EarthMass;
const sdp R_p = 1.0;
const sdp epsilon = 0.0;

//=======================================================================
// Grid parameters
//=======================================================================

const sdp dx_min = 0.0012;
const sdp dx_max = 0.002;
const sdp dy_min = 0.0012;
const sdp dy_max = 0.015;
const sdp dz_min = 0.0005;
const sdp dz_max = 0.002;

const int ngeomx = 1;
const int ngeomy = 3;
const int ngeomz = 0;

//=======================================================================
// Structures
//=======================================================================

struct body
{
  sdp m;
  sdp x;
  sdp y;
  //sdp z;
};

struct SymDisk
{
  sdp r;
  sdp p;
  sdp u;
  sdp v;
  sdp w;
};

struct hydr_ring
{
  sdp r[jmax];
  sdp p[jmax];
  sdp u[jmax];
  sdp v[jmax];
  sdp w[jmax];

  sdp x;
  sdp dx;
  sdp xvol;
  sdp xc;

  sdp y[jmax];
  sdp dy[jmax];
  sdp yvol[jmax];
  sdp yc[jmax];

  sdp z;
  sdp dz;
  sdp zvol;
  sdp zc;

  int i;
  int k;

  sdp rot_v;
  sdp res_v;
  int rot_j;
  int inc_j;
};

struct GPU_plan
{
  int id;

  int N_ring;

  int istart;
  int iblk;

  int kstart;
  int kblk;

  long int memsize;

  hydr_ring *rings;
  hydr_ring *h_rings;

  hydr_ring *lft;
  hydr_ring *rgh;
  hydr_ring *udr;
  hydr_ring *top;

  hydr_ring *cp_lft;
  hydr_ring *cp_rgh;

  SymDisk *val;

  dim3 sx_grid;
  dim3 sy_grid;
  dim3 sz_grid;

  dim3 t_grid;

  sdp *dt;
  sdp *h_dt;
  sdp *dt_1D;
  sdp *dt_2D;

  sdp *h_output;
  sdp *h_output1;
  sdp *h_output2;
  sdp *h_output3;
  sdp *h_output4;
  sdp *h_output5;

  sdp *d_output;
  sdp *d_output1;
  sdp *d_output2;
  sdp *d_output3;
  sdp *d_output4;
  sdp *d_output5;

  cudaEvent_t event;

  cudaStream_t stream;
};

//=======================================================================
// GPU control
//=======================================================================

const int startid = 0;
//const int block_per_device = iblk*(jblk/nDev+1)*kblk;

void P2P_all_enable(GPU_plan*);
void P2P_all_disable(GPU_plan*);
void syncstreams(GPU_plan*);
void syncdevices(GPU_plan*);
void syncallevents(GPU_plan*);
void syncallevents(int, GPU_plan*);
void reset_devices();

//=======================================================================
// Make grid
//=======================================================================

void grid(int, int, sdp, sdp, sdp*, sdp*, sdp*, int);
__host__ __device__ sdp get_vol(sdp, sdp, int);

//=======================================================================
// Utilities
//=======================================================================

void wait_f_r();
string int_to_string (int num);
string frame_num (int num);
string path_to_cwd();

//=======================================================================
// Reduction
//=======================================================================

__device__ void shift_round_reduc_max(int, int, sdp*);
__device__ void round_reduc_max(int, sdp*);
__device__ void round_reduc_sum(int, sdp*);
__device__ void bin_reduc_max(int, sdp*);
__device__ void bin_reduc_sum(int, sdp*);

//=======================================================================
// Simple functions
//=======================================================================

__device__ sdp min3(sdp, sdp, sdp);
__device__ sdp max3(sdp, sdp, sdp);

#endif
