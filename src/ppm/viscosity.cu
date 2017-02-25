
__device__ void derivs(sdp &first, sdp &second, int &n, sdp* xf, sdp* u)
{
  sdp x2, x3, h2, h3, A, B;

  x2 = xf[n]  -xf[n-1];
  x3 = xf[n+1]-xf[n-1];

  h2 = (u[n]  -u[n-1])/x2;
  h3 = (u[n+1]-u[n-1])/x3;

  A = (   h3-   h2)/(x3-x2);
  B = (x3*h2-x2*h3)/(x3-x2);

  second = 2.0*A;
  first  = second*x2 + B;
  
  return;
}

__device__ void derivs(sdp &first, sdp &second, int &n, sdp &x2, sdp &x3, sdp* u)
{
  sdp h2, h3, A, B;

  h2 = (u[n]  -u[n-1])/x2;
  h3 = (u[n+1]-u[n-1])/x3;

  A = (   h3-   h2)/(x3-x2);
  B = (x3*h2-x2*h3)/(x3-x2);

  second = 2.0*A;
  first  = second*x2 + B;
  
  return;
}

__device__ void derivs(sdp &first, sdp &second, sdp x2, sdp x3, sdp u2, sdp u3)
{
  sdp h2, h3, A, B;

  h2 = u2/x2;
  h3 = u3/x3;

  A = (   h3-   h2)/(x3-x2);
  B = (x3*h2-x2*h3)/(x3-x2);

  second = 2.0*A;
  first  = second*x2 + B;
  
  return;
}

__device__ sdp deriv_1st(sdp u1, sdp u2, sdp u3, sdp x1, sdp x2, sdp x3)
{
  sdp h2, h3, n2, n3;

  n2 = x2 - x1;
  n3 = x3 - x1;

  h2 = (u2-u1)/n2;
  h3 = (u3-u1)/n3;
  
  return (n3*h2-n2*h3)/(n3-n2);
}

__device__ sdp deriv_2nd(sdp u1, sdp u2, sdp u3, sdp x1, sdp x2, sdp x3)
{
  sdp h2, h3, n2, n3;

  n2 = x2 - x1;
  n3 = x3 - x1;

  h2 = (u2-u1)/n2;
  h3 = (u3-u1)/n3;
  
  return 2.f*(h3-h2)/(n3-n2);
}

__device__ sdp cross_deriv(sdp x1y1, sdp x2y1, sdp x1y2, sdp x2y2, sdp dx, sdp dy)
{
  return (x2y2 + x1y1 + x1y2 + x2y1)/(dx*dy);
}

//================================================================================
/*
__global__ void viscosity_r(smcell *cells, sdp dt, sdp alpha)
{
  int i, j, k;
  int N = blockIdx.x;
  i = threadIdx.x;
  j = blockIdx.y+6;
  k = blockIdx.z+6;

  __shared__ sdp u[arrsize], v[arrsize], w[arrsize], nu[arrsize];
  sdp R, rho, nr;
  sdp d1, d2, dn;
  sdp x2, x3;
  sdp ddu, ddv, ddw;

  rho = cells[N].r[i][j][k];
  R   = cells[N].xc[i];
  nr  = alpha*get_cs2_device(R)*cpow(R,1.5)*rho;

  nu [i] = nr;
  u  [i] = cells[N].u[i][j][k];
  v  [i] = cells[N].v[i][j][k]/R;
  w  [i] = cells[N].w[i][j][k];
  __syncthreads();

  if (i>0 && i<arrsize-1)
  {
    d1 = cells[N].xc[i-1];
    x2 = R - d1;
    x3 = cells[N].xc[i+1] - d1;

    derivs(dn, d2, i, x2, x3, nu);
    derivs(d1, d2, i, x2, x3, u);
    ddu = 2.0*nr*d2 + 2.0*dn*d1 + 2.0*nr*d1/R - 2.0*nr*u[i]/R/R;

    derivs(d1, d2, i, x2, x3, v);
    ddv = R*nr*d2 + R*dn*d1 + 3.0*nr*d1;

    derivs(d1, d2, i, x2, x3, w);
    ddw = nr*d2 + dn*d1 + nr*d1/R;

    cells[N].u[i][j][k] += dt*ddu/rho;
    cells[N].v[i][j][k] += dt*ddv/rho;
    cells[N].w[i][j][k] += dt*ddw/rho;
  }

  return;
}

__global__ void viscosity_p(smcell *cells, sdp dt, sdp alpha)
{
  int i, j, k;
  int N = blockIdx.x;
  j = threadIdx.x;
  k = blockIdx.y+6;
  i = blockIdx.z+6;

  __shared__ sdp u[arrsize], v[arrsize], w[arrsize], nu[arrsize];
  sdp R, rho, nr;
  sdp d1, d2, dn;
  sdp x2, x3;
  sdp ddu, ddv, ddw;

  rho = cells[N].r[i][j][k];
  R   = cells[N].xc[i];
  nr  = alpha*get_cs2_device(R)*cpow(R,1.5)*rho;

  nu [j] = nr;
  u  [j] = cells[N].u[i][j][k];
  v  [j] = cells[N].v[i][j][k]/R;
  w  [j] = cells[N].w[i][j][k];
  __syncthreads();

  if (j>0 && j<arrsize-1)
  {
    d1 = cells[N].yc[j-1];
    x2 = (cells[N].yc[j] - d1)*R;
    x3 = (cells[N].yc[j+1] - d1)*R;

    derivs(dn, d2, j, x2, x3, nu);
    derivs(d1, d2, j, x2, x3, v);
    ddv = 2.f*R*nr*d2 + 2.f*R*dn*d1;
    ddu = -2.f*nr*d1;

    derivs(d1, d2, j, x2, x3, w);
    ddw = nr*d2 + dn*d1;

    derivs(d1, d2, j, x2, x3, u);
    ddu += nr*d2 + dn*d1;
    ddv += 3.f*nr*d1/R + 2.f*dn*u[j]/R;

    cells[N].v[i][j][k] += dt*ddv/rho;
    cells[N].w[i][j][k] += dt*ddw/rho;
    cells[N].u[i][j][k] += dt*ddu/rho;
  }

  return;
}

__global__ void viscosity_z(smcell *cells, sdp dt, sdp alpha)
{
  int i, j, k;
  int N = blockIdx.x;
  k = threadIdx.x;
  i = blockIdx.y+6;
  j = blockIdx.z+6;

  __shared__ sdp u[arrsize], v[arrsize], w[arrsize], nu[arrsize];
  sdp R, rho, nr;
  sdp d1, d2, dn;
  sdp x2, x3;
  sdp ddu, ddv, ddw;

  rho = cells[N].r[i][j][k];
  R   = cells[N].xc[i];
  nr  = alpha*get_cs2_device(R)*cpow(R,1.5)*rho;

  nu [k] = nr;
  u  [k] = cells[N].u[i][j][k];
  v  [k] = cells[N].v[i][j][k]/R;
  w  [k] = cells[N].w[i][j][k];
  __syncthreads();

  if (k>0 && k<arrsize-1)
  {
    d1 = cells[N].zc[k-1];
    x2 = cells[N].zc[k] - d1;
    x3 = cells[N].zc[k+1] - d1;

    derivs(dn, d2, k, x2, x3, nu);
    derivs(d1, d2, k, x2, x3, w);
    ddw = 2.f*nr*d2 + 2.f*dn*d1; 

    derivs(d1, d2, k, x2, x3, u);
    ddu = nr*d2 + dn*d1;

    derivs(d1, d2, k, x2, x3, v);
    ddv = R*nr*d2 + R*dn*d1;

    cells[N].w[i][j][k] += dt*ddw/rho;
    cells[N].u[i][j][k] += dt*ddu/rho;
    cells[N].v[i][j][k] += dt*ddv/rho;
  }

  return;
}

__global__ void viscosity_all(smcell *cells, sdp dt, sdp alpha)
{
  int i, j, k;
  int N = blockIdx.x;

  sdp ddu, ddv, ddw, dn, d1, d2;

  sdp R, rho;

  __shared__ sdp u[arrsize], v[arrsize], w[arrsize], n[arrsize], x[arrsize];

  sdp n1, u1, v1, w1, x1;


  ///////////////////////////////////////////////////////////////////////////////////

  i = threadIdx.x;
  j = blockIdx.y+6;
  k = blockIdx.z+6;

  rho = cells[N].r[i][j][k];
  R   = cells[N].xc[i];

  u1 = cells[N].u[i][j][k];
  v1 = cells[N].v[i][j][k]/R;
  w1 = cells[N].w[i][j][k];
  n1 = alpha*get_cs2_device(R)*cpow(R,1.5)*rho;
  x1 = cells[N].xc[i];

  n[i] = n1;
  u[i] = u1;
  v[i] = v1;
  w[i] = w1;
  x[i] = x1;
  __syncthreads();

  if (i>6 && i<arrsize-6){

  derivs(dn, d2, i, x, n);
  derivs(d1, d2, i, x, u);
  ddu  = 2.0*(n1*d2 + dn*d1 + n1*d1/R - n1*u1/R/R);

  derivs(d1, d2, i, x, v);
  ddv  = R*n1*d2 + R*dn*d1 + 3.0*n1*d1;

  derivs(d1, d2, i, x, w);
  ddw  = n1*d2 + dn*d1 + n1*d1/R;

  cells[N].u[i][j][k] = u1 + dt*ddu/rho;
  cells[N].v[i][j][k] = v1*R + dt*ddv/rho;
  cells[N].w[i][j][k] = w1 + dt*ddw/rho;
  }
  __syncthreads();
  ///////////////////////////////////////////////////////////////////////////////////

  j = threadIdx.x;
  k = blockIdx.y+6;
  i = blockIdx.z+6;

  rho = cells[N].r[i][j][k];
  R   = cells[N].xc[i];

  u1 = cells[N].u[i][j][k];
  v1 = cells[N].v[i][j][k]/R;
  w1 = cells[N].w[i][j][k];
  n1 = alpha*get_cs2_device(R)*cpow(R,1.5)*rho;
  x1 = cells[N].yc[j];

  n[j] = n1;
  u[j] = u1;
  v[j] = v1;
  w[j] = w1;
  x[j] = x1;
  __syncthreads();

  if (j>6 && j<arrsize-6){

  derivs(dn, d2, j, x, n);
  derivs(d1, d2, j, x, v);
  ddv = 2.0*R*n1*d2 + 2.0*R*dn*d1;
  ddu =-2.0*n1*d1;

  derivs(d1, d2, j, x, w);
  ddw = n1*d2 + dn*d1;

  derivs(d1, d2, j, x, u);
  ddu += n1*d2 + dn*d1;
  ddv += 3.0*n1*d1/R + 2.0*dn*u1/R;

  cells[N].u[i][j][k] = u1 + dt*ddu/rho;
  cells[N].v[i][j][k] = v1*R + dt*ddv/rho;
  cells[N].w[i][j][k] = w1 + dt*ddw/rho;
  }
  __syncthreads();
  ///////////////////////////////////////////////////////////////////////////////////

  k = threadIdx.x;
  i = blockIdx.y+6;
  j = blockIdx.z+6;

  rho = cells[N].r[i][j][k];
  R   = cells[N].xc[i];

  u1 = cells[N].u[i][j][k];
  v1 = cells[N].v[i][j][k]/R;
  w1 = cells[N].w[i][j][k];
  n1 = alpha*get_cs2_device(R)*cpow(R,1.5)*rho;
  x1 = cells[N].zc[k];

  n[k] = n1;
  u[k] = u1;
  v[k] = v1;
  w[k] = w1;
  x[k] = x1;
  __syncthreads();

  if (k>6 && k<arrsize-6){

  derivs(dn, d2, k, x, n);
  derivs(d1, d2, k, x, w);
  ddw = 2.0*n1*d2 + 2.0*dn*d1; 

  derivs(d1, d2, k, x, u);
  ddu = n1*d2 + dn*d1;

  derivs(d1, d2, k, x, v);
  ddv = R*n1*d2 + R*dn*d1;

  cells[N].u[i][j][k] = u1 + dt*ddu/rho;
  cells[N].v[i][j][k] = v1*R + dt*ddv/rho;
  cells[N].w[i][j][k] = w1 + dt*ddw/rho;
  }
  __syncthreads();
  return;
}
*/
//================================================================================

__device__ void device_viscosity_r(int i, int lim, sdp dt, sdp R, sdp *x, sdp *cell_r, sdp *u, sdp *v, sdp *w, sdp *n)
{
  sdp rho, nr, d1, d2, dn;
  sdp ddu, ddv, ddw;

  if (i>=0 && i<lim)
  {
    rho = cell_r[i];
    nr = get_nu(R)*rho;
    //nr = get_cs2_device(R)*cpow(R,1.5)*rho;
    //nr *= alpha;

    n[i] = nr;
    v[i] /= R*R;
  }
  __syncthreads();

  if (i>=n_pad && i<lim-n_pad)
  {
    derivs(dn, d2, i, x, n);

    derivs(d1, d2, i, x, u);
    ddu = 2.0*nr*d2 + 2.0*dn*d1 + 2.0*nr*d1/R - 2.0*nr*u[i]/R/R;
    ddu = u[i] + dt*ddu/rho;

    #if ndim > 1
    derivs(d1, d2, i, x, v);
    ddv = R*nr*d2 + R*dn*d1 + 3.0*nr*d1;
    ddv = v[i]*R + dt*ddv/rho;
    #endif

    #if ndim == 3
    derivs(d1, d2, i, x, w);
    ddw = nr*d2 + dn*d1 + nr*d1/R;
    ddw = w[i] + dt*ddw/rho;
    #endif
  }
  __syncthreads();

  if (i>=n_pad && i<lim-n_pad)
  {
    u[i] = ddu;
    #if ndim > 1
    v[i] = ddv*R;
    #if ndim == 3
    w[i] = ddw;
    #endif
    #endif
  }

  return;
}

__device__ void device_viscosity_p(int j, int lim, sdp dt, sdp R, sdp *x, sdp *cell_r, sdp *u, sdp *v, sdp *w, sdp *n)
{
  sdp rho, nr, d1, d2, dn;
  sdp ddu, ddv, ddw;

  if (j>=0 && j<lim)
  {
    rho = cell_r[j];
    nr = get_nu(R)*rho;
  //nr  = get_cs2_device(R)*cpow(R,1.5)*rho;
  //nr *= alpha;

    n[j] = nr;
  }
  __syncthreads();

  if (j>=n_pad && j<lim-n_pad)
  {
    derivs(dn, d2, j, x, n);

    derivs(d1, d2, j, x, v);
    ddv = 2.0*nr*d2 + 2.0*dn*d1;
    //ddu =-2.0*nr*d1/R;
    ddu = 0.0;
    ddv = v[j] + dt*ddv/rho;

    #if ndim == 3
    derivs(d1, d2, j, x, w);
    ddw = nr*d2 + dn*d1;
    ddw = w[j] + dt*ddw/rho;
    #endif

    derivs(d1, d2, j, x, u);
    ddu += nr*d2 + dn*d1;
    //ddv += nr*d1/R + 2.0*dn*u[j]/R;
    ddu = u[j] + dt*ddu/rho;
  }
  __syncthreads();

  if (j>=n_pad && j<lim-n_pad)
  {
    v[j] = ddv;
    #if ndim == 3
    w[j] = ddw;
    #endif
    u[j] = ddu;
  }

  return;
}

__device__ void device_viscosity_z(int k, sdp dt, sdp R, sdp *x, sdp *cell_r, sdp *u, sdp *v, sdp *w, sdp *n)
{
  sdp nr, d1, d2, dn;
  sdp ddu, ddv, ddw;

  sdp rho = cell_r[k];
  nr = vis_nu*rho;
  //nr  = get_cs2_device(R)*cpow(R,1.5)*rho;
  //nr *= alpha;

  //if (dis>0.0171) nr *= ss_alpha;
  //else if (dis>0.0171/2.0) nr *= (0.0171/0.03 - ss_alpha)*(0.0171-dis)/(0.0171/2.0) + ss_alpha;
  //else nr *= 0.0171/0.03;

  n[k] = nr;
  __syncthreads();

  if (k>0 && k<arrsize-1)
  {
    derivs(dn, d2, k, x, n);
    derivs(d1, d2, k, x, w);
    ddw = 2.0*nr*d2 + 2.0*dn*d1; 

    derivs(d1, d2, k, x, u);
    ddu = nr*d2 + dn*d1;
    //ddw += nr*d1/R;

    derivs(d1, d2, k, x, v);
    ddv = nr*d2 + dn*d1;

    ddw = w[k] + dt*ddw/rho;
    ddu = u[k] + dt*ddu/rho;
    ddv = v[k] + dt*ddv/rho;
  }
  __syncthreads();

  if (k>0 && k<arrsize-1)
  {
    w[k] = ddw;
    u[k] = ddu;
    v[k] = ddv;
  }

  return;
}
