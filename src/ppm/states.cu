//=======================================================================

__device__ void states( int &n, int nmax, sdp hdt, int &axis,
                        sdp &rad, sdp &azi, sdp &pol, sdp &v_rot, body planet, 
                        sdp *al, sdp *da, sdp *a6, sdp *cs, sdp *csfac,
                        StateVar &S, sdp &flat,
                        sdp *r, sdp *p, sdp *u, sdp *v, sdp *w, sdp *xa0, sdp *dx0)
{
  // Compute parabolic coefficients and volume elements
  ParaConst par;
  paraset( n, 0, nmax, par, dx0,
           al, da, a6 ); //temporary variables
  //-----------------------------------------------------------------------

  // This subroutine takes the values of rho, u, and P at the left hand
  // side of the zone, the change accross the zone, and the parabolic 
  // coefficients, p6, u6, and rho6, and computes the left and right states
  // (integrated over the charachteristics) for each variable for input 
  // to the Riemann solver.
  //-----------------------------------------------------------------------
  sdp tmp, ftot;

  if (n>=2 && n<nmax-1)
  {
    if (axis==0)
      get_fx_bound(n, hdt, xa0[n], azi+hdt*v_rot, pol, tmp, ftot, planet, r, p, u, v, w);
    else if (axis==1)
      get_fy      (n, hdt, rad, xa0[n]+hdt*v_rot, pol, tmp, ftot, planet, r, p, u, v, w);
    else
      get_fz_bound(n, hdt, rad, azi+hdt*v_rot, xa0[n], tmp, ftot, planet, r, p, u, v, w);
  }
  __syncthreads();

  cs[n]    = hdt*csqrt(gam*p[n]/r[n])/dx0[n];
  if (axis==1) cs[n] /= rad;
  if (axis==2) 
  {
    if (ngeomz == 5) cs[n] /= rad;
  }
  csfac[n] = 1.0 - fourthd*cs[n];

  parabola(n, 0, nmax, flat, p, al, da, a6, par);
  if (n>=3 && n<nmax-2)
  {
    tmp  = al[n-1] + da[n-1] - cs[n-1]*(da[n-1] - csfac[n-1]*a6[n-1]);
    S.pl = cmax(smallp, tmp);
    tmp  = al[n]             + cs[n]  *(da[n]   + csfac[n]  *a6[n]);
    S.pr = cmax(smallp, tmp);  
  }
  __syncthreads();

  parabola(n, 0, nmax, flat, r, al, da, a6, par);
  if (n>=3 && n<nmax-2)
  {
    tmp  = al[n-1] + da[n-1] - cs[n-1]*(da[n-1] - csfac[n-1]*a6[n-1]);
    S.rl = cmax(smallr, tmp);
    tmp  = al[n]             + cs[n]  *(da[n]   + csfac[n]  *a6[n]);
    S.rr = cmax(smallr, tmp);  
  }
  __syncthreads();

  parabola(n, 0, nmax, flat, u, al, da, a6, par);
  if (n>=3 && n<nmax-2)
  {
    tmp  = al[n-1] + da[n-1] - cs[n-1]*(da[n-1] - csfac[n-1]*a6[n-1]);
    S.ul = tmp + hdt*ftot;
    tmp  = al[n]             + cs[n]  *(da[n]   + csfac[n]  *a6[n]);
    S.ur = tmp + hdt*ftot;
  }
  __syncthreads();

  return;
}
