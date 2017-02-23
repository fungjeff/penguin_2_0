__device__ sdp get_flux(int &n, int nmax, sdp *flux, sdp *al, sdp *da, sdp *a6, sdp &dxf, sdp &dxf2, sdp &fr, bool &delC)
{
  if (n>=6 && n<nmax-5)
  {
    if(delC) flux[n] = (al[n-1] + da[n-1] - dxf*(da[n-1] - dxf2*a6[n-1]))*fr;
    else     flux[n] = (al[n]             - dxf*(da[n]   + dxf2*a6[n]  ))*fr;
  }
  __syncthreads();
  return flux[n]-flux[n+1];
}

//========================================================================

__device__ void remap(int &n, int nmax, int &axis, sdp &rad, sdp &azi, sdp &pol, body &planet,
                      sdp *al, sdp *da, sdp *a6,
                      sdp *r, sdp *p, sdp *u, sdp *v, sdp *w, sdp *e, sdp *q, sdp xa0, sdp xa, sdp *dx,
                      sdp dvol, sdp dvol0, sdp flat=0.0)
{
  // Compute parabolic coefficients and volume elements
  ParaConst par;
  paraset( n, 3, nmax-3, par, dx,
           al, da, a6 ); //temporary variables
  //-----------------------------------------------------------------------

  // Remap mass, momentum, and energy from the updated lagrangian grid
  // to the fixed Eulerian grid, using piecewise parabolic functions.
  //-----------------------------------------------------------------------
  sdp lvol, dxn, dxnm;

  if (axis==0)
  {
    if (ngeomx==0) lvol = xa - xa0;
    else if (ngeomx==1)
    {
      lvol  = xa - xa0;
      lvol *= (xa0+0.5*lvol);
    }
    else if (ngeomx==2)
    {
      lvol  = xa - xa0;
      lvol *= xa0*xa + third*(lvol*lvol);
    }
  }
  else if (axis==1)
  {
    if (ngeomy==0)      lvol = xa - xa0;
    else if (ngeomy==3) lvol = (xa - xa0) * rad;
    else if (ngeomy==4) lvol = (xa - xa0) * rad;
  }
  else
  {
    if (ngeomz==0) lvol = xa - xa0;
    else if (ngeomz==5) lvol = (ccos(xa0) - ccos(xa)) * rad;
  }

  bool delC;
  sdp dxf, dxf2, fr;
  
  dxf = xa - xa0;
  if(dxf >= 0.0) delC = true;
  else           delC = false;

  parabola(n, 3, nmax-3, flat, r, al, da, a6, par);

  sdp *flux;
  if (n>=6 && n<nmax-5)
  {
    dxn = dx[n];
    dxnm = dx[n-1];
  }
  flux = dx;
  __syncthreads();

  if (n>=6 && n<nmax-5)
  {
    if(delC)
    {
      dxf  = 0.5*dxf/dxnm;
      dxf2 = 1.0 - fourthd*dxf;
      fr   = (al[n-1] + da[n-1] - dxf*(da[n-1] - dxf2*a6[n-1]))*lvol;
      flux[n] = fr;
    }
    else
    {
      dxf  = 0.5*dxf/dxn;
      dxf2 = 1.0 + fourthd*dxf;
      fr   = (al[n]             - dxf*(da[n]   + dxf2*a6[n]  ))*lvol;
      flux[n] = fr;
    }
  }
  __syncthreads();

  sdp m, invm, flux_n;
  if (n>=6 && n<nmax-6)
  {
    m = r[n] * dvol;
    lvol = (m + fr - flux[n+1])/dvol0;

    r[n] = cmax(smallr, lvol);

    invm = 1.0/(r[n]*dvol0);
  }

  //-----------------------------------------------------------------------------------------------------------
  parabola(n, 3, nmax-3, flat, u, al, da, a6, par);

  flux_n = get_flux(n, nmax, flux, al, da, a6, dxf, dxf2, fr, delC);

  if (n>=6 && n<nmax-6) u[n] = (u[n]*m + flux_n)*invm;
  //-----------------------------------------------------------------------------------------------------------
  parabola(n, 3, nmax-3, flat, v, al, da, a6, par);

  flux_n = get_flux(n, nmax, flux, al, da, a6, dxf, dxf2, fr, delC);

  if (n>=6 && n<nmax-6) v[n] = (v[n]*m + flux_n)*invm;
  //-----------------------------------------------------------------------------------------------------------
  parabola(n, 3, nmax-3, flat, w, al, da, a6, par);

  flux_n = get_flux(n, nmax, flux, al, da, a6, dxf, dxf2, fr, delC);

  if (n>=6 && n<nmax-6) w[n] = (w[n]*m + flux_n)*invm;
  //-----------------------------------------------------------------------------------------------------------
  #if EOS == 0
  parabola(n, 3, nmax-3, flat, q, al, da, a6, par);

  flux_n = get_flux(n, nmax, flux, al, da, a6, dxf, dxf2, fr, delC);

  if (n>=6 && n<nmax-6)
  {
    q[n] = (q[n]*m + flux_n)*invm;
    p[n] = cmax(smallp, r[n]*q[n]);
  }
  #elif EOS == 1
  parabola(n, 3, nmax-3, flat, q, al, da, a6, par);

  flux_n = get_flux(n, nmax, flux, al, da, a6, dxf, dxf2, fr, delC);

  if (n>=6 && n<nmax-6)
  {
    q[n] = (q[n]*m + flux_n)*invm;
    p[n] = cmax(smallp, pow(r[n],gam)*q[n]);
  }

  #elif EOS == 2
  parabola(n, 3, nmax-3, flat, q, al, da, a6, par);

  flux_n = get_flux(n, nmax, flux, al, da, a6, dxf, dxf2, fr, delC);

  if (n>=6 && n<nmax-6)
  {
    q[n] = (q[n]*m + flux_n)*invm;
    p[n] = cmax(smallp, r[n]*q[n]);
  }
  #endif
/*
  #else
  

  __syncthreads();
  parabola(n, 3, nmax-3, flat, e, al, da, a6, par);

  flux_n = get_flux(n, nmax, flux, al, da, a6, dxf, dxf2, fr, delC);

  if (n>=6 && n<nmax-6) 
  {
    e[n] = (e[n]*m + flux_n)*invm;
    if (axis==0)
    {
      p[n] = gamm*r[n]*( e[n] - 0.5*((u[n]*u[n])+(v[n]*v[n]/rad/rad)+(w[n]*w[n])) );
    }
    else if (axis==1)
    {
      p[n] = gamm*r[n]*( e[n] - 0.5*((u[n]*u[n])+(v[n]*v[n])+(w[n]*w[n])) );
    }
    else
    {
      p[n] = gamm*r[n]*( e[n] - 0.5*((u[n]*u[n])+(v[n]*v[n])+(w[n]*w[n])) );
    }
    p[n] = cmax(p[n],smallp);
  }
*/
  //#endif
/*
  #ifdef NOT_ISO
  sdp KE;

  parabola(n, 3, nmax-3, flat, e, al, da, a6, par);

  flux_n = get_flux(n, nmax, flux, al, da, a6, dxf, dxf2, fr, delC);

  if (n>=6 && n<nmax-6) e[n] = (e[n]*m + flux_n)*invm;
  __syncthreads();

  //-----------------------------------------------------------------------------------------------------------
  parabola(n, 3, nmax-3, flat, q, al, da, a6, par);

  flux_n = get_flux(n, nmax, flux, al, da, a6, dxf, dxf2, fr, delC);

  if (n>=6 && n<nmax-6)
  {
    q[n] = (q[n]*m + flux_n)*invm;

    // If flow is highly supersonic remap on internal energy, else on total E

    //KE = 0.5*((u[n]*u[n])+(v[n]*v[n])+(w[n]*w[n]));

    //if (KE/q[n] < 100.0 ) KE = e[n] - KE;
    //else                  KE = q[n];
    

    p[n] = cmax(smallp, r[n]*q[n]);
  }
  #else
  parabola(n, 3, nmax-3, flat, q, al, da, a6, par);
  flux_n = get_flux(n, nmax, flux, al, da, a6, dxf, dxf2, fr, delC);

  if (n>=6 && n<nmax-6)
  {
    q[n] = (q[n]*m + flux_n)*invm;
    p[n] = cmax(smallp, r[n]*q[n]);
  }
  #endif
*/
  __syncthreads();
  return;
}
