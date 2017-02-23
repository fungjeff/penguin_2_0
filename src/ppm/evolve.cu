//=======================================================================================

__device__ void evolve( int &n, int nmax, sdp &dt, int &axis,
                        sdp &rad, sdp &azi, sdp &pol, sdp &v_rot, body &planet,
                        sdp *vfac,
                        sdp *umid, sdp *pmid, bool &riemann_success,
                        sdp *r, sdp *p, sdp *u, sdp *v, sdp *w, sdp *e, sdp *q,
                        sdp *xa0, sdp* dx0, sdp &dvol0, sdp &xa, sdp* dx, sdp &dvol)
{
  sdp dtdm, delx;
  sdp gra1, fto1, gra2, fto2;
  sdp dum, xcl;

  vfac[n] = xa0[n] + 0.5*dx0[n];
  //if (!riemann_success) printf("riemann failed at %f\n",xa0[n]);//pmid[n] = p[n];
  __syncthreads();

  if (n>=3 && n<nmax-2)
  {
    if      (axis==0) delx = dt*umid[n];
    else if (axis==1) 
    {
      if (ngeomy > 2) delx = dt*(umid[n]/rad-v_rot-FrRot);
      else            delx = dt*umid[n];
    }
    else
    {
      if (ngeomz == 5) delx = dt*umid[n]/rad;
      else             delx = dt*umid[n];
    }
    dtdm    = dt / (r[n] * dvol0);
    vfac[n] = xa0[n] + delx;
  }
  __syncthreads();
/*
  if (n==6 && blockIdx.x==0 && axis==1 && rad<1.3) printf("%i %e\n", n, umid[n]);
  if (n==7 && blockIdx.x==0 && axis==1 && rad<1.3) printf("%i %e\n", n, umid[n]);
  if (n==8 && blockIdx.x==0 && axis==1 && rad<1.3) printf("%i %e\n\n", n, umid[n]);

  if (n==6 && blockIdx.x==234 && axis==1 && rad<1.3) printf("%i %e\n", n, umid[n]);
  if (n==7 && blockIdx.x==234 && axis==1 && rad<1.3) printf("%i %e\n", n, umid[n]);
  if (n==8 && blockIdx.x==234 && axis==1 && rad<1.3) printf("%i %e\n\n\n", n, umid[n]);
*/
  if (n>=3 && n<nmax-3)
  {
    dx[n] = vfac[n+1] - vfac[n];
    xa    = vfac[n];
    //if (dx[n]<0.0)
    //{
    //  printf(" Serious Warning at (%i, %f, %f) with (%f, %f) along axis %i.\n",n,vfac[n+1],vfac[n],r[n],p[n]/r[n],axis);
    //}
  }
  __syncthreads();

  // Calculate forces using coordinates at t and at t+dt, note that
  // fictitious forces at t+dt depend on updated velocity, but we ignore this

  if (n>=3 && n<nmax-3)
  {
    xcl = xa + 0.5*dx[n];
    if (axis==0)
    {
      get_fx(n, 0.0, rad, azi, pol, gra1, fto1, planet, r, p, u, v, w);
      get_fx(n,  dt, xcl, azi+dt*v_rot, pol, gra2, fto2, planet, r, p, u, v, w);
    }
    else if (axis==1)
    {
      get_fy(n, 0.0, rad, azi, pol, gra1, fto1, planet, r, p, u, v, w);
      get_fy(n,  dt, rad, xcl+dt*v_rot, pol, gra2, fto2, planet, r, p, u, v, w);
    }
    else
    {
      get_fz(n, 0.0, rad, azi, pol, gra1, fto1, planet, r, p, u, v, w);
      get_fz(n,  dt, rad, azi+dt*v_rot, xcl, gra2, fto2, planet, r, p, u, v, w);
/*
      if (nudr==0)
      {
        if (n<6) {gra1 *= -1.0; fto1 *= -1.0; gra2 *= -1.0; fto2 *= -1.0;}
      }
      if (ntop==0)
      {
      //if (n>=arrsize-6) {gra1 *= -1.0; fto1 *= -1.0; gra2 *= -1.0; fto2 *= -1.0;}
      }
*/
    }
  }
  __syncthreads();

  if (n>=3 && n<nmax-2)
  { 
    umid[n] *= pmid[n];
    if (axis==0)
    {
      if (ngeomx==0)
      {
        dvol = dx[n];
        vfac[n] = 1.0;
      }
      else if (ngeomx==1)
      {
        dvol = dx[n]*(xa+0.5*dx[n]);
        vfac[n] = 0.5*(xa + xa0[n]);
      }
      else if (ngeomx==2)
      {
        dvol = dx[n]*(xa*(xa+dx[n])+dx[n]*dx[n]*third);
        vfac[n] = (xa-xa0[n])*(third*(xa-xa0[n])+xa0[n])+(xa0[n]*xa0[n]);
      }
    }
    else if (axis==1)
    {
      if (ngeomy==0)
      {
        dvol = dx[n];
        vfac[n] = 1.0;
      }
      else if (ngeomy==3)
      {
        dvol = dx[n]*rad;
        vfac[n] = 1.0;
      }
      else if (ngeomy==4)
      {
        dvol = dx[n]*rad;
        vfac[n] = 1.0;
      }
    }
    else
    {
      if (ngeomz==0)
      {
        dvol = dx[n];
        vfac[n] = 1.0;
      }
      else if (ngeomz==5)
      {
        delx = xa - xa0[n];
        dum  = ccos(xa);
        dvol = ( dum-ccos(xa+dx[n]) )*rad;
        if(delx == 0.0) vfac[n] = csin(xa);
        else vfac[n] = ( ccos(xa0[n])-dum )/delx;
      }
    }
  }
  __syncthreads();

  if (n>=3 && n<nmax-3)
  {
    #if EOS == 0
    q[n] = p[n]/r[n];
    #elif EOS == 1
    q[n] = p[n]/pow(r[n],gam);
    #endif

    r[n] = cmax( r[n]*(dvol0/dvol), smallr);

    dum  = u[n];
    delx = 0.5*dt*(fto1+fto2);
    delx += -dtdm*(pmid[n+1]-pmid[n])*0.5*(vfac[n+1]+vfac[n]);
    u[n] = dum + delx;

    #if EOS == 2
    //e[n] = e[n] - dtdm*(vfac[n+1]*umid[n+1] - vfac[n]*umid[n]) + 0.5*dt*(dum*gra1 + u[n]*gra2);

    if (axis==0)
    {
      q[n] = e[n] - dtdm*(vfac[n+1]*umid[n+1] - vfac[n]*umid[n]) - 0.5*((u[n]*u[n])+(v[n]*v[n]/xcl/xcl)) - potential(xcl, azi, pol, planet);
    }
    else if (axis==1)
    {
      q[n] = e[n] - dtdm*(vfac[n+1]*umid[n+1] - vfac[n]*umid[n]) - 0.5*(u[n]*u[n]) - potential(rad, xcl, pol, planet);
    }
    else
    {
      q[n] = e[n] - dtdm*(vfac[n+1]*umid[n+1] - vfac[n]*umid[n]) - 0.5*(u[n]*u[n]) - potential(rad, azi, xcl, planet);
    }
    q[n] = cmax(gamm*q[n],smallp);

    #endif
  }
  __syncthreads();

  return;
}

//=======================================================================================
