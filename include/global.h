#ifndef GLOBAL_H
#define GLOBAL_H

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

#define flat_flag
//#define bary_flag
#define EOS 0                    // 0:isothermal 1:isentropic 2:energy(adiabatic)

#define plnt_flag 1
#define opac_flag 0
#define kill_flag 0
#define visc_flag 0
#define BlHo_flag 0

#define dump_flag 1
#define file_flag 0

//=======================================================================
// Disk parameters
//=======================================================================

const sdp p_alpha = 1.5;
const sdp p_beta = 0.0;
const sdp ss_alpha = 0.01;
const sdp sc_h = 0.035;
const sdp vis_nu = 1.0e-5;

//=======================================================================
// Planet parameters
//=======================================================================

const sdp M_p = 1.0*EarthMass;
const sdp R_p = 1.0;
const sdp epsilon = 0.5*sc_h*R_p * 0.5*sc_h*R_p;
const sdp FrRot = 1.0;

//=======================================================================
// Hydro parameters
//=======================================================================

const sdp gam = 1.0;
const sdp gamm = gam - 1.0;
const sdp gamfac2 = gam + 1.0;
const sdp gamfac1 = gamfac2/gam/2.0;
const sdp gamz = gamm/gam/2.0;
const sdp courant = 0.5;

const int nlft = 3;
const int nrgh = 3;
const int nbac = 2;
const int nfrn = 2;
const int nudr = 3;
const int ntop = 3;

const sdp endtime = 10.0*twopi;
const sdp tmovie = 1.0*twopi;

const int ncycend = 1000000000;
const int nprin = 50;

//=======================================================================
// Grid parameters
//=======================================================================

const sdp xmin = 0.4;
const sdp xmax = 1.8;
const sdp ymin = 0.0;
const sdp ymax = twopi;
const sdp zmin = hpi-0.065;//0.065;//0.088;//0.05;
const sdp zmax = hpi;

const int ngeomx = 1;
const int ngeomy = 3;
const int ngeomz = 0;

const int grspx = 1;
const int grspy = 0;
const int grspz = 0;

const sdp dx_min = 0.0012;
const sdp dx_max = 0.002;
const sdp dy_min = 0.0012;
const sdp dy_max = 0.015;
const sdp dz_min = 0.0005;
const sdp dz_max = 0.002;

using namespace std;

#define cmax(x,y) max(x,y)
#define cmin(x,y) min(x,y)
#define cabs(x) fabs(x)
#define csqrt(x) sqrt(x)
#define cexp(x) exp(x)
#define clog(x) log(x)
#define csin(x) sin(x)
#define ccos(x) cos(x)
#define ctan(x) tan(x)
#define cpow(x,y) pow(x,y)
#define sign(x) (( x > 0 ) - ( x < 0 ))

//=======================================================================
// GPU control
//=======================================================================

const int startid = 0;
const int nDev = 1;

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
