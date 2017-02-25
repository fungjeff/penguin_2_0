#ifndef VARIABLE_TYPES_H
#define VARIABLE_TYPES_H

typedef double sdp;
typedef double2 sdp2;

//=======================================================================
// GPU Arch and Grid Dim
//=======================================================================

#define ndim 2

const int n_pad = 6;                      //number of boundary zones on each side: 6 for ppm

const int arrsize = 128;                  //for optimal speed: arrsize = 32*(integer)
const int realarr = arrsize - 2*n_pad;

const int imax = 850;                     //for optimal speed: imax = realarr*(integer)
const int jmax = 2000;                    //same for jmax and kmax
const int kmax = 1;

//=======================================================================
// Structures
//=======================================================================

struct body
{
  sdp m;
  sdp x;
  sdp y;
  //sdp z;
  sdp vx;
  sdp vy;
  sdp fx;
  sdp fy;
  sdp rs;
};

struct ParaConst
{
  sdp c0;
  sdp c1;
  sdp c2;
  sdp c3;
  sdp c4;
};

struct StateVar
{
  sdp rl;
  sdp pl;
  sdp ul;
  sdp rr;
  sdp pr;
  sdp ur;
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

struct result
{
  sdp q_ratio;
  sdp aspect_ratio;
  sdp alpha;
  sdp start_t;
  sdp end_t;

  sdp r[imax][jmax][kmax];
  sdp p[imax][jmax][kmax];
  sdp u[imax][jmax][kmax];
  sdp v[imax][jmax][kmax];
  sdp w[imax][jmax][kmax];

  sdp x[imax];
  sdp dx[imax];

  sdp y[jmax];
  sdp dy[jmax];

  sdp z[kmax];
  sdp dz[kmax];
};

#endif
