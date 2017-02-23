//#########################################################################################

void set_size_for_data()
{
  zro.resize(imax);
  zpr.resize(imax);
  zux.resize(imax);
  zuy.resize(imax);
  zuz.resize(imax);

  for (int i=0; i<imax; i++)
  {
    zro[i].resize(jmax);
    zpr[i].resize(jmax);
    zux[i].resize(jmax);
    zuy[i].resize(jmax);
    zuz[i].resize(jmax);

    for (int j=0; j<jmax; j++)
    {
      zro[i][j].resize(kmax);
      zpr[i][j].resize(kmax);
      zux[i][j].resize(kmax);
      zuy[i][j].resize(kmax);
      zuz[i][j].resize(kmax);
    }
  }
  return;
}

//#########################################################################

void init_grid()
{
// Sod shock tube problem (a whimpy test) in 1, 2, or 3 dimensions
// 24jan92 blondin
//=======================================================================

//--------------------------------------------------------------------------------
// Set up geometry and boundary conditions of grid
//
// Boundary condition flags : nleft, nright
//   = 0  :  reflecting boundary condition
//   = 1  :  inflow/outflow boundary condition (zero gradients)
//   = 2  :  fixed inflow boundary condition (values set by dinflo, pinflo, etc.)
//   = 3  :  periodic (nmax+1 = nmin; nmin-1 = nmax)
//
// Geometry flag : ngeom                         |  Cartesian:
//   = 0  :  planar                              |    gx = 0, gy = 0, gz = 0   (x,y,z)
//   = 1  :  cylindrical radial                  |  Cylindrical:
//   = 2  :  spherical   radial             3D= {     gx = 1, gy = 3, gz = 0   (s,phi,z)
//   = 3  :  cylindrical angle                   |
//   = 4  :  spherical azimu angle (phi)         |  Spherical:
//   = 5  :  spherical polar angle (theta)       |    gx = 2, gy = 4, gz = 5   (r,phi,theta)

  for (int i = 0; i<imax; i++)
  {
    zxa[i]=0.0;
    zdx[i]=0.0;
  }
  for (int j = 0; j<kmax; j++)
  {
    zya[j]=0.0;
    zdy[j]=0.0;
  }
  for (int k = 0; k<kmax; k++)
  {
    zza[k]=0.0;
    zdz[k]=0.0;
  }

  grid(0, imax, xmin, xmax, zxa, zxc, zdx, grspx);
  grid(0, jmax, ymin, ymax, zya, zyc, zdy, grspy);
  grid(0, kmax, zmin, zmax, zza, zzc, zdz, grspz);

  return;
}

void copy_grid(GPU_plan *set)
{
  int i, j, k;
  for (int v=0; v<nDev; v++)
  {
    for (int n=0; n<set[v].N_ring; n++)
    {  
      i = set[v].h_rings[n].i;
      zxa[i] = set[v].h_rings[n].x;
      zdx[i] = set[v].h_rings[n].dx;
      zxc[i] = set[v].h_rings[n].xc;

      k = set[v].h_rings[n].k;
      zza[k] = set[v].h_rings[n].z;
      zdz[k] = set[v].h_rings[n].dz;
      zzc[k] = set[v].h_rings[n].zc;
    }
  }

  for (int j = 0; j<jmax; j++)
  {
    zya[j] = set[0].h_rings[0].y[j];
    zdy[j] = set[0].h_rings[0].dy[j];
    zyc[j] = set[0].h_rings[0].yc[j];
  }

  return;
}

void update_grid(GPU_plan *set)
{
  int i, j, k, j0;
  for (int v=0; v<nDev; v++)
  {
    for (int n=0; n<set[v].N_ring; n++)
    {  
      i = set[v].h_rings[n].i;
      k = set[v].h_rings[n].k;

      for (int j = 0; j<jmax; j++)
      {
        j0 = j - set[v].h_rings[n].rot_j;
        if (j0>=jmax) j0 -= jmax;
        if (j0<0) j0 += jmax;
        zro[i][j][k] = set[v].h_rings[n].r[j0];
        zpr[i][j][k] = set[v].h_rings[n].p[j0];
        zux[i][j][k] = set[v].h_rings[n].u[j0];
        zuy[i][j][k] = set[v].h_rings[n].v[j0];
        zuz[i][j][k] = set[v].h_rings[n].w[j0];
      }
    }
  }
  return;
}
//#########################################################################################
