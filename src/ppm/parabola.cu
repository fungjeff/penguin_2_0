//======================================================================
/*
__device__ void parabola(int &n, int nmin, int nmax, sdp flat,
                         sdp *val, sdp *al, sdp &da, sdp *a6,
                         ParaConst &par)
{
// Colella and Woodward, JCompPhys 54, 174-201 (1984) eq 1.5-1.8,1.10
//
// parabola calculates the parabolas themselves. call paraset first
// for a given grid-spacing to set the constants, which can be reused
// each time parabola is called.
//
// flatening coefficients are calculated externally in flaten. 
// nmin/nmax are indicies over which the parabolae are calculated
//-----------------------------------------------------------------------
  sdp t1, t2, a, ar, la, da_r, da_l;

  if (n>=nmin+1 && n<nmax-1)
  {
    a = val[n];
    da_r = val[n+1] - a;
    da_l = a - val[n-1];

    t1 = par.c3 * da_r + par.c4 * da_l;
    if (t1 < 0.0) t1 = -min3( -t1, 2.0*cabs(da_l), 2.0*cabs(da_r) );
    else          t1 =  min3(  t1, 2.0*cabs(da_l), 2.0*cabs(da_r) );

    if(da_l*da_r < 0.0) t1 = 0.0; 
    a6[n] = t1;
  }
  __syncthreads();

  if (n>=nmin+1 && n<nmax-2)
  {
    ar = a + par.c0*da_r + par.c1*a6[n+1] + par.c2*t1;
    al[n+1] = ar;
  }
  __syncthreads();

  if (n>=nmin+2 && n<nmax-2)
  {
    t1 = flat * a;
    t2 = 1.0 - flat;
    ar = t1 + t2 * ar;
    la = t1 + t2 * al[n];

    t1 = ar - la;

    t2 = t1*t1;
    t1 = t1*( 6.0 * (a - 0.5*(ar + la) ) );

    if ( (ar-a)*(a-la) <= 0.0)
    {
      ar = a;
      la = a;
    }
    if (t2 <  t1) la = 3.0 * a - 2.0 * ar;
    if (t2 < -t1) ar = 3.0 * a - 2.0 * la;
  
    da    = ar - la;
    a6[n] = 6.0 * (a - 0.5*(ar + la));
    al[n] = la;
  }
  __syncthreads();
  return;
}
*/
//======================================================================

__device__ void parabola(int &n, int nmin, int nmax, sdp flat,
                         sdp *val, sdp *al, sdp *da, sdp *a6,
                         ParaConst &par)
{
// Colella and Woodward, JCompPhys 54, 174-201 (1984) eq 1.5-1.8,1.10
//
// parabola calculates the parabolas themselves. call paraset first
// for a given grid-spacing to set the constants, which can be reused
// each time parabola is called.
//
// flatening coefficients are calculated externally in flaten. 
// nmin/nmax are indicies over which the parabolae are calculated
//-----------------------------------------------------------------------
  sdp t1, t2, a, ar, la, da_r, da_l;

  if (n>=nmin+1 && n<nmax-1)
  {
    a = val[n];
    da_r = val[n+1] - a;
    da_l = a - val[n-1];

    t1 = par.c3 * da_r + par.c4 * da_l;
    if (t1 < 0.0) t1 = -min3( -t1, 2.0*cabs(da_l), 2.0*cabs(da_r) );
    else          t1 =  min3(  t1, 2.0*cabs(da_l), 2.0*cabs(da_r) );

    if(da_l*da_r < 0.0) t1 = 0.0; 
    a6[n] = t1;
  }
  __syncthreads();

  if (n>=nmin+1 && n<nmax-2)
  {
    ar = a + par.c0*da_r + par.c1*a6[n+1] + par.c2*t1;
    al[n+1] = ar;
  }
  __syncthreads();

  if (n>=nmin+2 && n<nmax-2)
  {
    t1 = flat * a;
    t2 = 1.0 - flat;
    ar = t1 + t2 * ar;
    la = t1 + t2 * al[n];

    t1 = ar - la;

    t2 = t1*t1;
    t1 = t1*( 6.0 * (a - 0.5*(ar + la) ) );

    if ( (ar-a)*(a-la) <= 0.0)
    {
      ar = a;
      la = a;
    }
    if (t2 <  t1) la = 3.0 * a - 2.0 * ar;
    if (t2 < -t1) ar = 3.0 * a - 2.0 * la;
  
    da[n] = ar - la;
    a6[n] = 6.0 * (a - 0.5*(ar + la));
    al[n] = la;
  }
  __syncthreads();
  return;
}

//#######################################################################


__device__ void paraset(int &n, int nmin, int nmax, ParaConst &par,
                        sdp *dx, sdp *a, sdp *ai, sdp *b)
{
  sdp dx_n, an, ain, bn, bi, c, ci, d;

  if (n>=nmin && n<nmax-1)
  {
    dx_n = dx[n];

    an  = dx_n + dx[n+1];
    ain = 1.0/an;
    bn  = an + dx_n;
    bi  = 1.0/bn;
    c   = an + dx[n+1];
    ci  = 1.0/c;

    a[n] = an;
    ai[n] = ain;
    b[n] = bn;
  }
  __syncthreads();

  if (n>=nmin+1 && n<nmax-2)
  {
    d = 1.0/(a[n-1] + a[n+1]);
    par.c0 =  dx_n * ain * ( 1.0 + 2.0*d*dx[n+1]*( a[n-1]*bi - a[n+1]*ci ) );
    par.c1 = -d*dx_n   *a[n-1]*bi;
    par.c2 =  d*dx[n+1]*a[n+1]*ci;
  }

  if (n>=nmin+1 && n<nmax-1)
  {
    d = dx_n/(a[n-1] + dx[n+1] );
    par.c3 = d * b[n-1] * ain;
    par.c4 = d * c      * ai[n-1];
  }
  __syncthreads();
  return;
}
