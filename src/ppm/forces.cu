
//====================================================================

__device__ sdp potential(sdp &rad, sdp &azi, sdp &z, body &p)
{
  #if plnt_flag==1
  sdp cosfac = ccos(azi-p.y);
  sdp stm = 1.0/(1.0+p.m);
  sdp plm = p.m*stm;
  sdp plx = p.x*stm;
  sdp stx = p.x*plm;

  sdp Rp = rad*rad + plx*plx - 2.0*plx*rad*cosfac + z*z;
  Rp = csqrt(Rp+epsilon);

  sdp Rs = csqrt(rad*rad + stx*stx + 2.0*stx*rad*cosfac + z*z);
  return -plm/Rp - stm/Rs;
  #else
  return -1.0/csqrt(rad*rad+z*z);
  #endif
}
//====================================================================
__device__ sdp star_planet_grav_rad(sdp rad, sdp &azi, sdp pol, body &p, sdp dt)
{
  sdp cosfac = ccos(azi-(p.y+dt*p.vy));
  sdp sinpol = csin(pol);

  #ifdef bary_flag
  sdp stm = 1.0/(1.0+p.m);
  sdp plm = p.m*stm;
  sdp plx = p.x*stm;
  sdp stx = p.x*plm;

  sdp Rp = rad*rad + plx*plx - 2.0*plx*rad*cosfac*sinpol;
  Rp = csqrt(Rp+epsilon);

  sdp Rs = csqrt(rad*rad + stx*stx + 2.0*stx*rad*cosfac*sinpol);

  return -plm*(rad-plx*cosfac*sinpol)/(Rp*Rp*Rp) - stm*(rad+stx*cosfac*sinpol)/(Rs*Rs*Rs);
  #else
  sdp stm = 1.0/(1.0+p.m);
  sdp plm = p.m*stm;
  sdp plx = p.x;

  sdp Rp = rad*rad + plx*plx - 2.0*plx*rad*cosfac*sinpol;
  Rp = csqrt(Rp+epsilon);

  return -plm*(rad-plx*cosfac*sinpol)/(Rp*Rp*Rp) - stm/(rad*rad) - plm*sinpol*cosfac/(plx*plx);
  #endif
}

__device__ sdp star_planet_grav_azi(sdp rad, sdp &azi, sdp pol, body &p, sdp dt)
{
  sdp cosfac = ccos(azi-(p.y+dt*p.vy));
  sdp sinfac = csin(azi-(p.y+dt*p.vy));
  sdp sinpol = csin(pol);
  rad /= sinpol;  

  #ifdef bary_flag
  sdp stm = 1.0/(1.0+p.m);
  sdp plm = p.m*stm;
  sdp plx = p.x*stm;
  sdp stx = p.x*plm;

  sdp Rp = rad*rad + plx*plx - 2.0*plx*rad*cosfac*sinpol;
  Rp = csqrt(Rp+epsilon);

  sdp Rs = csqrt(rad*rad + stx*stx + 2.0*stx*rad*cosfac*sinpol);

  return -plm*plx*sinfac/(Rp*Rp*Rp) + stm*stx*sinfac/(Rs*Rs*Rs);
  #else
  sdp stm = 1.0/(1.0+p.m);
  sdp plm = p.m*stm;
  sdp plx = p.x;

  sdp Rp = rad*rad + plx*plx - 2.0*plx*rad*cosfac*sinpol;
  Rp = csqrt(Rp+epsilon);

  return -plm*plx*sinfac/(Rp*Rp*Rp) + plm*sinpol*sinfac/plx/plx;
  #endif
}

__device__ sdp star_planet_grav_pol(sdp rad, sdp &azi, sdp pol, body &p, sdp dt)
{
  sdp cosfac = ccos(azi-(p.y+dt*p.vy));
  sdp cospol = ccos(pol);
  sdp sinpol = csin(pol);
  #ifdef bary_flag
  sdp stm = 1.0/(1.0+p.m);
  sdp plm = p.m*stm;
  sdp plx = p.x*stm;
  sdp stx = p.x*plm;

  sdp Rp = rad*rad + plx*plx - 2.0*plx*rad*cosfac*sinpol;
  Rp = csqrt(Rp+epsilon);

  sdp Rs = csqrt(rad*rad + stx*stx + 2.0*stx*rad*cosfac*sinpol);

  return plm*plx*cosfac*cospol/(Rp*Rp*Rp) - stm*stx*cosfac*cospol/(Rs*Rs*Rs);
  #else
  sdp stm = 1.0/(1.0+p.m);
  sdp plm = p.m*stm;
  sdp plx = p.x;

  sdp Rp = rad*rad + plx*plx - 2.0*plx*rad*cosfac*sinpol;
  Rp = csqrt(Rp+epsilon);

  return plm*plx*cosfac*cospol/(Rp*Rp*Rp) - plm*cosfac*cospol/(sinpol*plx*plx);
  #endif
}
//====================================================================
__device__ sdp star_planet_grav_rad_cyl(sdp rad, sdp azi, sdp z, body &p, sdp dt)
{
  sdp cosfac = ccos(azi-(p.y+dt*p.vy));
  sdp stm = 1.0/(1.0+p.m);
  sdp plm = p.m*stm;
  sdp plx = p.x*stm;
  sdp stx = p.x*plm;

  sdp Rp = rad*rad + plx*plx - 2.0*plx*rad*cosfac + z*z;
  Rp = csqrt(Rp+epsilon);

  sdp Rs = csqrt(rad*rad + stx*stx + 2.0*stx*rad*cosfac + z*z);
  return -plm*(rad-plx*cosfac)/(Rp*Rp*Rp) - stm*(rad+stx*cosfac)/(Rs*Rs*Rs);
}

__device__ sdp star_planet_grav_azi_cyl(sdp rad, sdp azi, sdp z, body &p, sdp dt)
{
  sdp cosfac;
  sdp sinfac;
  sincos(azi-(p.y+dt*p.vy), &sinfac, &cosfac);
  sdp stm = 1.0/(1.0+p.m);
  sdp plm = p.m*stm;
  sdp plx = p.x*stm;
  sdp stx = p.x*plm;

  sdp Rp = rad*rad + plx*plx - 2.0*plx*rad*cosfac + z*z;
  Rp = csqrt(Rp+epsilon);

  sdp Rs = csqrt(rad*rad + stx*stx + 2.0*stx*rad*cosfac + z*z);
  return -plm*plx*sinfac/(Rp*Rp*Rp) + stm*stx*sinfac/(Rs*Rs*Rs);
}

__device__ sdp planet_grav_azi_cyl(sdp rad, sdp azi, sdp z, body &p, sdp dt)
{
  sdp cosfac;
  sdp sinfac;
  sincos(azi-(p.y+dt*p.vy), &sinfac, &cosfac);
  sdp stm = 1.0/(1.0+p.m);
  sdp plm = p.m*stm;
  sdp plx = p.x*stm;
  sdp stx = p.x*plm;

  sdp Rp = rad*rad + plx*plx - 2.0*plx*rad*cosfac + z*z;
  Rp = csqrt(Rp+epsilon);

  sdp Rs = csqrt(rad*rad + stx*stx + 2.0*stx*rad*cosfac + z*z);
  return -plm*plx*sinfac/(Rp*Rp*Rp);
}

__device__ sdp star_planet_grav_pol_cyl(sdp rad, sdp azi, sdp z, body &p, sdp dt)
{
  sdp cosfac = ccos(azi-(p.y+dt*p.vy));
  sdp stm = 1.0/(1.0+p.m);
  sdp plm = p.m*stm;
  sdp plx = p.x*stm;
  sdp stx = p.x*plm;

  sdp Rp = rad*rad + plx*plx - 2.0*plx*rad*cosfac + z*z;
  Rp = csqrt(Rp+epsilon);

  sdp Rs = csqrt(rad*rad + stx*stx + 2.0*stx*rad*cosfac + z*z);
  return -plm*z/(Rp*Rp*Rp) - stm*z/(Rs*Rs*Rs);
}

//=======================================================================
__device__ void get_fx_bound(int &n, sdp dt, sdp rad, sdp azi, sdp pol,
                       sdp &grav, sdp &ftot, body &planet,
                       sdp *r, sdp *p, sdp *u, sdp *v, sdp *w)
{
  if (ngeomx==0)  // CARTESIAN
  {
    grav = 0.0;
    ftot = grav;
  }
    
  else if (ngeomx==1)   // CYLINDRICAL R
  {
    #if plnt_flag==1
    grav = star_planet_grav_rad_cyl(rad, azi, pol, planet, dt);
    #else
    sdp dis = csqrt(rad*rad+pol*pol);
    grav = -rad/dis/dis/dis;
    #endif

    ftot  = grav;
    ftot += cpow(0.5*(v[n]+v[n-1]),2)/rad/rad/rad;
    //if (n==6 && blockIdx.x==0) printf("%.12f + %.12f = %.12f\n",grav, ftot-grav, ftot);
  }  
  else if (ngeomx==2)  // SPHERICAL R
  {
    #if plnt_flag==1
    grav = star_planet_grav_rad(rad, azi, pol, planet, dt);
    #else
    grav = -1.0/rad/rad;
    #endif

    ftot = grav;
    ftot += cpow(0.5*(w[n]+w[n-1]),2)/rad/rad/rad;
    ftot += cpow(0.5*(v[n]+v[n-1]),2)/rad/cpow(rad*csin(pol),2);
  }
  return;
}

__device__ void get_fx(int &n, sdp dt, sdp rad, sdp azi, sdp pol,
                       sdp &grav, sdp &ftot, body &planet,
                       sdp *r, sdp *p, sdp *u, sdp *v, sdp *w)
{
  if (ngeomx==0)  // CARTESIAN
  {
    grav = 0.0;
    ftot = grav;
  }
  else if (ngeomx==1)   // CYLINDRICAL R
  {
    #if plnt_flag==1
    grav = star_planet_grav_rad_cyl(rad, azi, pol, planet, dt);
    #else
    sdp dis = csqrt(rad*rad+pol*pol);
    grav = -rad/dis/dis/dis;
    #endif

    ftot  = grav;
    ftot += v[n]*v[n]/rad/rad/rad;
  } 
  else if (ngeomx==2)  // SPHERICAL R
  {
    #if plnt_flag==1
    grav = star_planet_grav_rad(rad, azi, pol, planet, dt);
    #else
    grav = -1.0/rad/rad;
    #endif

    ftot = grav;
    ftot += w[n]*w[n]/rad/rad/rad;
    ftot += v[n]*v[n]/rad/cpow(rad*csin(pol),2);
  }
  return;
}

__device__ void get_fy(int &n, sdp dt, sdp rad, sdp azi, sdp pol,
                       sdp &grav, sdp &ftot, body &planet,
                       sdp *r, sdp *p, sdp *u, sdp *v, sdp *w)
{
  if (ngeomy==0)  // CARTESIAN
  {
    grav = 0.0;
    ftot = grav;
  }
  else if (ngeomy==3)  // CYLINDRICAL PHI
  {
    #if plnt_flag==1
    grav = star_planet_grav_azi_cyl(rad, azi, pol, planet, dt);
    #else
    grav = 0.0;
    #endif

    ftot = grav;
  } 
  else if (ngeomy==4)  // SPHERICAL PHI
  {
    #if plnt_flag==1
    grav = star_planet_grav_azi(rad, azi, pol, planet, dt);
    #else
    grav = 0.0;
    #endif

    ftot = grav;
    //ftot += (-2.0*u[n]/rad) * (w[n] + v[n]/ctan(pol));
  }
  return;
}

__device__ void get_fz_bound(int &n, sdp dt, sdp rad, sdp azi, sdp pol,
                             sdp &grav, sdp &ftot, body &planet,
                             sdp *r, sdp *p, sdp *u, sdp *v, sdp *w)
{
  if (ngeomz==0)  // CARTESIAN OR CYLINDRICAL
  {
    #if plnt_flag==1
    grav = star_planet_grav_pol_cyl(rad, azi, pol, planet, dt);
    #else
    sdp dis = csqrt(rad*rad+pol*pol);
    grav = -pol/dis/dis/dis;
    #endif

    ftot = grav;
    
    #if kill_flag==2
    ftot += damping(rad,0.5*(u[n]+u[n+1]));
    #endif
  }
  else if (ngeomz==5)  // SPHERICAL THETA
  {
    #if plnt_flag==1
    grav = star_planet_grav_pol(rad, azi, pol, planet, dt);
    #else
    grav = 0.0;
    #endif

    sdp rad_cyl = rad*csin(pol);
    ftot = grav;
    ftot += (cpow(0.5*(w[n]+w[n-1]),2)*ccos(pol))/rad_cyl/rad_cyl/rad_cyl;

    #if kill_flag==2
    ftot += damping(rad,0.5*(u[n]+u[n+1]));
    #endif
  }
  return;
}

__device__ void get_fz(int &n, sdp dt, sdp rad, sdp azi, sdp pol,
                       sdp &grav, sdp &ftot, body &planet,
                       sdp *r, sdp *p, sdp *u, sdp *v, sdp *w)
{
  if (ngeomz==0)  // CARTESIAN OR CYLINDRICAL
  {
    #if plnt_flag==1
    grav = star_planet_grav_pol_cyl(rad, azi, pol, planet, dt);
    #else
    sdp dis = csqrt(rad*rad+pol*pol);
    grav = -pol/dis/dis/dis;
    #endif

    ftot = grav;
    
    #if kill_flag==2
    ftot += damping(rad,u[n]);
    #endif
  }
  else if (ngeomz==5)  // SPHERICAL THETA
  {
    #if plnt_flag==1
    grav = star_planet_grav_pol(rad, azi, pol, planet, dt);
    #else
    grav = 0.0;
    #endif

    sdp rad_cyl = rad*csin(pol);
    ftot = grav;
    ftot += (w[n]*w[n]*ccos(pol))/rad_cyl/rad_cyl/rad_cyl;

    #if kill_flag==2
    ftot += damping(rad,u[n]);
    #endif
  }
  return;
}
