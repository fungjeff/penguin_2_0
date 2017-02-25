//================================================================================
__device__ void ppmlr(sdp &rad, sdp &azi, sdp &pol, sdp v_rot, sdp FrRot,
                      sdp *r, sdp *p, sdp *u, sdp *v, sdp *w, sdp *e,
                      sdp *xa0, sdp *dx0, sdp &dvol0, 
                      int nmax, sdp &dt, int axis, body &planet)
{
  __shared__ sdp q[arrsize];
  __shared__ sdp dx[arrsize];
  __shared__ sdp umid[arrsize], pmid[arrsize];
  __shared__ sdp tmp1[arrsize];

  int n = threadIdx.x;

  q[n]    = 0.0;
  dx[n]   = 0.0;
  umid[n] = 0.0;
  pmid[n] = 0.0;
  tmp1[n] = 0.0;

  sdp flat = 0.0;
  StateVar S;
  __syncthreads();

  // Calculate flattening coefficients for smoothing near shocks
  #ifdef flat_flag
  flatten( n, nmax, p, u, tmp1, flat );
  #endif

  // Integrate parabolae to get input states for Riemann problem
  states( n, nmax, 0.5*dt, axis, rad, azi, pol, v_rot, planet,
          q, dx, tmp1, umid, pmid, //temporary variables
          S, flat,
          r, p, u, v, w, xa0, dx0 );

  // Call pol Riemann solver to obtain pol zone face averages, umid and pmid
  bool riemann_success;
  riemann( n, nmax, S, umid, pmid, riemann_success );

  sdp dvol = dvol0;
  sdp xa   = xa0[n];
  // do lagrangian update using umid and pmid
  evolve( n, nmax, dt, axis, rad, azi, pol, v_rot, FrRot, planet,
          tmp1,//temporary variables
          umid, pmid, riemann_success,
          r, p, u, v, w, e, q,
          xa0, dx0, dvol0, xa, dx, dvol );

  //umid and pmid can be used as temporary variables

  // remap onto original Eulerian grid
  remap( n, nmax, axis, rad, azi, pol, planet,
         tmp1, umid, pmid, //temporary variables
         r, p, u, v, w, e, q, xa0[n], xa, dx, dvol, dvol0, flat );

  return;
}

//================================================================================
