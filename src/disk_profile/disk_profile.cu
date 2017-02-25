#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "variable_types.h"
#include "global.h"

sdp get_rho(sdp r)
{
  sdp n, rho;
  #if EOS == 0
  n = -p_alpha;
  rho = pow(r, n);
  #else
  if (ndim==3)
  {
    n = -p_alpha * (2.0/gamfac2);
    rho = pow(r, n);
  }
  else
  {
    n = -p_alpha;
    rho = pow(r, n);
  }
  #endif
  return rho;
}

sdp get_drho(sdp r)
{
  sdp n = -p_alpha;
  sdp rho = pow(r, n);
  sdp drho = n*rho/r;

  return drho;
}

__host__ __device__ sdp get_cs2(sdp r)
{
  return sc_h*sc_h*pow(r,-p_beta);
}

sdp get_dcs2(sdp r)
{
  return -p_beta*get_cs2(r)/r;
}

__host__ __device__ sdp get_h(sdp r)
{
  return sqrt(get_cs2(r)*r*r*r);
}

sdp get_P(sdp r)
{
  #if EOS == 0
  return get_cs2(r)*get_rho(r);
  #else
  return get_cs2(r)*pow(get_rho(r),gam)/gam;
  #endif
}

sdp get_dP_dr(sdp r)
{
  sdp p1 = get_P(r+1.27245e-8);
  sdp p2 = get_P(r-1.27245e-8);
  return (p1-p2)/2.5449e-8;
}

sdp get_verfac(sdp r, sdp z)
{
  sdp cs2 = get_cs2(r);
  sdp rr = sqrt(r*r+z*z);
  return exp((1.0/rr - 1.0/r)/cs2);
}

sdp get_dverfac(sdp r, sdp z)
{
  sdp cs2 = get_cs2(r);
  sdp dcs2 = get_dcs2(r);
  sdp rr = sqrt(r*r+z*z);
  sdp f = exp((1.0/rr - 1.0/r)/cs2);
  return ( f/(cs2*r*r) ) * ( (1.0 - pow(r/rr,3)) - (r*dcs2/cs2)*(r/rr - 1.0) );
}

sdp get_rho(sdp r, sdp z)
{
  #if EOS == 0
  return get_rho(r)*get_verfac(r,z);
  
  #else
  sdp h = sqrt(get_cs2(r)*r*r*r);
  sdp z_0 = sqrt(2.0*h*h*pow(r,-p_alpha*(2.0/gamfac2)*gamm)/gamm);
  if (z < z_0)
  {
    return pow( pow( get_rho(r),gamm ) - (gamm/(r*get_cs2(r))) * (1.0-r/sqrt(r*r+z*z)), 1.0/gamm );
  }
  else
  {
    return pow(10.0, log10(pow( pow( get_rho(r),gamm ) - (gamm/(r*get_cs2(r))) * (1.0-r/sqrt(r*r+z_0*z_0)), 1.0/gamm )) - 6.0*(z-z_0)/h );
  }
  #endif
}

sdp get_drho(sdp r, sdp z)
{
  sdp r1 = get_rho(r+1.0e-8,z);
  sdp r2 = get_rho(r-1.0e-8,z);
  return (r1-r2)/2.0e-8;
}

sdp get_P(sdp r, sdp z)
{
  #if EOS == 0
  return get_cs2(r)*get_rho(r,z);
  #else
  return get_cs2(r)*pow(get_rho(r,z),gam)/gam;
  #endif
}

sdp get_dP_dr(sdp r, sdp z)
{
  sdp p1 = get_P(r+1.27e-8,z);
  sdp p2 = get_P(r-1.27e-8,z);
  return (p1-p2)/2.54e-8;
}

sdp get_dP_s(sdp r, sdp pol)
{
  sdp dr = 1.27e-7;
  sdp sinpol = sin(pol);
  sdp cospol = cos(pol);
  sdp r1 = (r+dr)*sinpol;
  sdp z1 = (r+dr)*cospol;
  sdp r2 = (r-dr)*sinpol;
  sdp z2 = (r-dr)*cospol;
  return (get_P(r1,z1) - get_P(r2,z2)) / (2.0*dr);
}

sdp get_dln_Omega(sdp r, sdp z)
{
  sdp r1 = r+1.27e-7;
  sdp r2 = r-1.27e-7;
  sdp o1 = log(sqrt(pow(r1*r1+z*z,-1.5) + get_dP_dr(r1,z)/get_rho(r1,z)/r));
  sdp o2 = log(sqrt(pow(r2*r2+z*z,-1.5) + get_dP_dr(r2,z)/get_rho(r2,z)/r));
  return (o1-o2)/(log(r1)-log(r2));
}

__host__ __device__ sdp get_nu(sdp r)
{
  return vis_nu*pow(r,p_alpha);
}

sdp get_viscous_vr(sdp r)
{
  #if visc_flag == 1
  return -1.5*get_nu(r)/r;
  #else
  return 0.0;
  #endif
}

sdp get_viscous_vr(sdp r, sdp z)
{
  #if visc_flag == 1
  return -1.5*get_nu(r)/r;
  #else
  return 0.0;
  #endif

}

//#########################################################################################
