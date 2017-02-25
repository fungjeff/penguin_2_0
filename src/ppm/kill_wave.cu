//================================================================================

__global__ void kill_wave(smcell *cells, SymDisk *val, sdp dt, sdp rmin, sdp rmax)
{
  int n = blockIdx.x;
  int i = threadIdx.x+6;
  int j = blockIdx.y+6;
  int k = blockIdx.z+6;
  int ii;

  sdp f, vk;
  sdp d_kill = 0.5*sc_h;

  sdp r = cells[n].xc[i];

  sdp inner = rmin+d_kill;
  sdp outer = rmax-d_kill;

  if      (r<inner) 
  {
    ii = cells[n].i[i] + imax*cells[n].k[k];
    vk = val[ii].v;
    f = (inner-r)/d_kill;
  }
  else if (r>outer)
  {
    ii = cells[n].i[i] + imax*cells[n].k[k];
    vk = val[ii].v;
    f = (r-outer)/d_kill;
  }
  else return;

  //f *= f;
  f *= dt*cpow(r,-1.5);

  cells[n].r[i][j][k] += f * ( val[ii].r - cells[n].r[i][j][k] );
  cells[n].p[i][j][k] += f * ( val[ii].p - cells[n].p[i][j][k] );
  cells[n].u[i][j][k] += f * ( val[ii].u - cells[n].u[i][j][k] );
  cells[n].v[i][j][k] += f * ( vk        - cells[n].v[i][j][k] );
  cells[n].w[i][j][k] += f * (           - cells[n].w[i][j][k] );

  return;
}
