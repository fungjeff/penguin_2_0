#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "variable_types.h"
#include "global.h"
#include "disk_profile.h"
#include "output.h"

__host__ __device__ sdp get_L(body &p, sdp FrRot)
{
  sdp vy = p.vy+FrRot;
  return vy*p.x*p.x;
}

__host__ __device__ sdp get_E(body &p, sdp FrRot)
{
  sdp vy = p.vy+FrRot;
  return (p.vx*p.vx + vy*vy*p.x*p.x)/2.0 - 1.0/p.x;
}

__host__ __device__ sdp get_ecc(body &p, sdp FrRot)
{
  return sqrt(1.0 + (2.0*get_E(p,FrRot)*get_L(p,FrRot)*get_L(p,FrRot)));
}

sdp set_M_p(sdp t)
{
  if (t < t_growth)
    return max(M_p*sin(pi*t/(2.0*t_growth)), 0.01*EarthMass);
  else
    return M_p;
}

void init_planet(body &planet, sdp &FrRot)
{
  planet.m = set_M_p(0.0);
  planet.x = R_p;
  planet.y = pi;
  planet.vx = 0.0;
  planet.vy = pow(planet.x,-1.5);
  #if FrRot_flag == 1
  FrRot = planet.vy;
  #else
  FrRot = 0.0;
  #endif
  planet.vy -= FrRot;
  planet.rs = rs_fac*get_h(planet.x);

  return;
}

void planet_forces(body &planet, sdp FrRot, GPU_plan *set, sdp simtime)
{
  planet.m = set_M_p(simtime);
  #if plnt_flag==2
  planet.fx = Sigma_0*GPU_output_reduction(set, planet, 0.0, 8)/planet.m - 1.0/planet.x/planet.x + planet.x*cpow(planet.vy+FrRot, 2);
  planet.fy = Sigma_0*GPU_output_reduction(set, planet, 0.0, 9)/planet.m;
  #else
  planet.fx = -1.0/planet.x/planet.x + planet.x*cpow(planet.vy+FrRot, 2);
  planet.fy = 0.0;
  #endif
  return;
}

void kick_drift_kick(body &planet, sdp &FrRot, GPU_plan *set, sdp simtime, sdp dt, bool kick)
{
  sdp old_rad;  
  if (!kick)
  {
    old_rad = planet.x;
    planet.x += dt*planet.vx;
    planet.y += dt*planet.vy;
    planet.vy = ((planet.vy+FrRot)*old_rad*old_rad/planet.x/planet.x)-FrRot;
    if (planet.y<0.0)    planet.y += twopi;
    if (planet.y>=twopi) planet.y -= twopi;
    planet.rs = rs_fac*get_h(planet.x);
  }
  else
  {
    planet.vx += dt*planet.fx;
    planet.vy += FrRot + dt*planet.fy/planet.x/planet.x;
    #if FrRot_flag == 1
    FrRot = planet.vy;
    #else
    FrRot = 0.0;
    #endif
    planet.vy -= FrRot;
  }

  return;
}

//#########################################################################################
