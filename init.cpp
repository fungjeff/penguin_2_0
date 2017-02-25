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

// If any dimension is angular, multiply coordinates by pi...

// set time and cycle counters

// Set up grid coordinates and parabolic coefficients

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

  //for (int k = 0; k<jmax; k++) cout << k << " " << zya[k] << " " << zdy[k] << " " << endl;
  //                                  << get_rho(xmin,zzc[k]) << " " << get_viscous_vr(xmin, zzc[k])/(-1.5*vis_nu/xmin) << endl;
  //wait_f_r();
//  find_places(imax, zxa);

  return;
}

void copy_grid(GPU_plan *set)
{
  int i, k;
  for (int v=0; v<nDev; v++)
  {
    for (int n=0; n<set[v].N_ring; n++)
    {
      i = set[v].h_rings[n].i;
      k = set[v].h_rings[n].k;
      if (i>=0 && i<imax)
      {
        zxa[i] = set[v].h_rings[n].x;
        zxc[i] = set[v].h_rings[n].xc;
        zdx[i] = set[v].h_rings[n].dx;
      }
      if (k>=0 && k<kmax)
      {
        zza[k] = set[v].h_rings[n].z;
        zzc[k] = set[v].h_rings[n].zc;
        zdz[k] = set[v].h_rings[n].dz;
      }
    }
  }
  for (int j=0; j<jmax; j++)
  {
    zya[j] = set[0].h_rings[0].y[j];
    zyc[j] = set[0].h_rings[0].yc[j];
    zdy[j] = set[0].h_rings[0].dy[j];
  }

  return;
}

void init_den(SymDisk *val)
{

//======================================================================
// Set up parameters from the problem (Sod shock tube)
/*
  sdp dright = 0.1;
  sdp pright = K_gam*pow(dright,gam);
  sdp dleft  = 1.0;
  sdp pleft  = K_gam*pow(dleft,gam);
*/
//=======================================================================

// initialize grid:

  double r, z;

  for (int i = 0; i<imax; i++)
  {
    for (int k = 0; k<kmax; k++)
    {
      if (ngeomz==5)
      {
        r = zxc[i]*sin(zzc[k]);
        z = zxc[i]*cos(zzc[k]);
      }
      else if (ngeomz==0)
      {
        r = zxc[i];
        z = zzc[k];
      }

      if (ndim==3)
      {
        val[i+imax*k].r = get_rho(r,z);
        val[i+imax*k].p = get_P(r,z);
      }
      else
      {
        val[i+imax*k].r = get_rho(r);
        val[i+imax*k].p = get_P(r);
      }
    }
  }

  return;
}

//#########################################################################################

void init_speed(SymDisk *val)
{
  double r, z, vk2;
  int i, k;
  double dP;

  for (i=0; i<imax; i++)
  {
    for (k=0; k<kmax; k++)
    {
      if (ndim==2)
      {
        r = zxc[i];
        z = 0.0;
        vk2 = 1.0/r;
        dP = get_dP_dr(zxc[i]);
      }
      else if (ngeomz==5)
      {
        vk2 = 1.0/zxc[i];
        r = zxc[i]*sin(zzc[k]);
        z = zxc[i]*cos(zzc[k]);
        dP = get_dP_s(zxc[i],zzc[k]);
      }
      else if (ngeomz==0)
      {
        r = zxc[i];
        z = zzc[k];
        dP = get_dP_dr(zxc[i], zzc[k]);
        vk2 = r*r*pow(r*r+z*z,-1.5);
      }

      vk2 += zxc[i]*dP/val[i+imax*k].r;
      val[i+imax*k].v = sqrt(vk2);

      if (ndim==3) val[i+imax*k].u = get_viscous_vr(r, z);
      else         val[i+imax*k].u = get_viscous_vr(r);
      val[i+imax*k].w = 0.0;
    }
  }
  return;
}

//#########################################################################################

void init_cells_val(hydr_ring &ring, int i, int k, SymDisk *val)
{
  int idx = i+imax*k;

  ring.k  = k;
  ring.z  = zza[k];
  ring.dz = zdz[k];
  ring.zc = ring.z + 0.5*ring.dz;

  ring.i  = i;
  ring.x  = zxa[i];
  ring.dx = zdx[i]; 
  ring.xc = ring.x + 0.5*ring.dx;
  
  for (int j=0; j<jmax; j++)
  {
    ring.y[j]  = zya[j];
    ring.dy[j] = zdy[j];
    ring.yc[j] = ring.y[j] + 0.5*ring.dy[j];

    ring.r[j] = val[idx].r;
    ring.p[j] = val[idx].p;
    ring.u[j] = val[idx].u;
    ring.v[j] = val[idx].v;
    ring.w[j] = 0.0;
  }

  ring.rot_j = 0;

  return;
}

//#########################################################################################

void init_cells_vol(hydr_ring &ring)
{
  ring.xvol = get_vol(ring.x, ring.dx, 0);
  ring.zvol = get_vol(ring.z, ring.dz, 2);
  for (int j = 0; j<jmax; j++) ring.yvol[j] = get_vol(ring.y[j], ring.dy[j], 1);

  return;
}

//#########################################################################################
void init_cells(GPU_plan *set, SymDisk *val)
{
  int m;

  for (int n=0; n<nDev; n++)
  {
    for (int k=0; k<set[n].kblk; k++)
    {
      for (int i=0; i<set[n].iblk; i++)
      {      
        m = i + set[n].iblk * k;

        init_cells_val(set[n].h_rings[m], i+set[n].istart, k+set[n].kstart, val);
        init_cells_vol(set[n].h_rings[m]);
      }
    }
  }
  return;
}

//#########################################################################################

void init_bound(hydr_ring *lft, hydr_ring *rgh, hydr_ring *udr, hydr_ring *top)
{
  double r, z, rr;
  double rho, P, dP, vk;
  int ib;

  if (nlft>=3)
  {
    for (int i = 0; i<n_pad; i++)
    {
      for (int k = 0; k<kmax; k++)
      {
        ib = i + n_pad*k;
        lft[ib].x = zxa[0]-zdx[0]*(double)(n_pad-i);
        lft[ib].dx = zdx[0];
        lft[ib].xc = lft[ib].x + 0.5*lft[ib].dx;
        lft[ib].xvol = get_vol( lft[ib].x , lft[ib].dx , 0 );

        lft[ib].z = zza[k];
        lft[ib].dz = zdz[k];
        lft[ib].zc = zzc[k];
        lft[ib].zvol = get_vol( zza[k] , zdz[k] , 2 );

        rr  = lft[ib].xc;

        if (ndim == 2)
        {
          r = rr;
          z = 0.0;

          rho = get_rho(r);
          P = get_P(r);
          dP = get_dP_dr(r);
          vk  = 1.0/r;
        }
        else if (ngeomz==5)
        {
          r = rr*sin(zzc[k]);
          z = rr*cos(zzc[k]);
          dP = get_dP_s(rr,zzc[k]);
          vk  = 1.0/rr;
        }
        else if (ngeomz==0)
        {
          r = rr;
          z = zzc[k];
          dP = get_dP_dr(r,z);
          vk = r*r*pow(r*r+z*z,-1.5);
        }
        vk += rr*dP/rho;
        vk  = sqrt(vk);

        for (int j = 0; j<jmax; j++)
        {
          lft[ib].r[j] = rho;
          lft[ib].p[j] = P;
          #if ndim==3
          lft[ib].u[j] = get_viscous_vr(r, z);
          #else
          lft[ib].u[j] = get_viscous_vr(r);
          #endif
          lft[ib].v[j] = vk;
          lft[ib].w[j] = 0.0;
          lft[ib].y[j] = zya[j];
          lft[ib].dy[j] = zdy[j];
          lft[ib].yc[j] = zyc[j];
          lft[ib].yvol[j] = get_vol( zya[j] , zdy[j] , 1 );
        }

        #if FARGO_flag == 1
        lft[ib].rot_v = vk - 1.0*lft[ib].xc;
        #elif FARGO_flag == 2
        lft[ib].rot_v = (1.0 - 1.0)*lft[ib].xc;
        #else
        lft[ib].rot_v = 0.0;
        #endif
        lft[ib].rot_j = 0;
      }
    }
  }

  if (nrgh==3)
  {
    for (int i = 0; i<n_pad; i++)
    {
      for (int k = 0; k<kmax; k++)
      {
        ib = i + n_pad*k;
        rgh[ib].x = zxa[imax-1] + zdx[imax-1]*(double)(i+1);
        rgh[ib].dx = zdx[imax-1];
        rgh[ib].xc = rgh[ib].x + 0.5*rgh[ib].dx;
        rgh[ib].xvol = get_vol( rgh[ib].x , rgh[ib].dx , 0 );

        rgh[ib].z = zza[k];
        rgh[ib].dz = zdz[k];
        rgh[ib].zc = zzc[k];
        rgh[ib].zvol = get_vol( zza[k] , zdz[k] , 2 );

        rr  = rgh[ib].xc;

        if (ndim == 2)
        {
          r = rr;
          z = 0.0;

          rho = get_rho(r);
          P = get_P(r);
          dP = get_dP_dr(r);
          vk  = 1.0/r;
        }
        else if (ngeomz==5)
        {
          r = rr*sin(zzc[k]);
          z = rr*cos(zzc[k]);

          rho = get_rho(r,z);
          P = get_P(r,z);
          dP = get_dP_s(rr,zzc[k]);
          vk  = 1.0/rr;
        }
        else if (ngeomz==0)
        {
          r = rr;
          z = zzc[k];

          rho = get_rho(r,z);
          P = get_P(r,z);
          dP = get_dP_dr(r,z);
          vk = r*r*pow(r*r+z*z,-1.5);
        }
        vk += rr*dP/rho;
        vk  = sqrt(vk);

        for (int j = 0; j<jmax; j++)
        {
          rgh[ib].r[j] = rho;
          rgh[ib].p[j] = P;
          #if ndim == 3
          rgh[ib].u[j] = get_viscous_vr(r, z);
          #else
          rgh[ib].u[j] = get_viscous_vr(r);
          #endif
          rgh[ib].v[j] = vk;
          rgh[ib].w[j] = 0.0;
          rgh[ib].y[j] = zya[j];
          rgh[ib].dy[j] = zdy[j];
          rgh[ib].yc[j] = zyc[j];
          rgh[ib].yvol[j] = get_vol( zya[j] , zdy[j] , 1 );
        }

        #if FARGO_flag == 1
        rgh[ib].rot_v = vk - 1.0*rgh[ib].xc;
        #elif FARGO_flag == 2
        rgh[ib].rot_v = (1.0 - 1.0)*rgh[ib].xc;
        #else
        rgh[ib].rot_v = 0.0;
        #endif
        rgh[ib].rot_j = 0;
      }
    }
  }

  #if ndim == 3
  if (nudr==3)
  {
    for (int k = 0; k<n_pad; k++)
    {
      for (int i = 0; i<imax; i++)
      { 
        ib = k + n_pad*i;
        udr[ib].z = zza[0]-zdz[0]*(double)(n_pad-k);
        udr[ib].dz = zdz[0];
        udr[ib].zc = udr[ib].z + 0.5*udr[ib].dz;  
        udr[ib].zvol = get_vol( udr[ib].z , udr[ib].dz , 2 );

        udr[ib].x = zxa[i];
        udr[ib].dx = zdx[i];
        udr[ib].xc = zxc[i];
        udr[ib].xvol = get_vol( zxa[i] , zdx[i] , 0 );

        rr  = zxc[i];

        if (ngeomz==5)
        {
          z = rr*cos(udr[ib].zc);
          r = sqrt(rr*rr - z*z);
          dP = get_dP_s(rr, udr[ib].zc);
        }
        else if (ngeomz==0)
        {
          r = rr;
          z = udr[ib].zc;
          dP = get_dP_dr(r,z);
        }
        rho = get_rho(r,z);
        P = get_P(r,z);
        vk  = 1.0/rr;
        vk += rr*dP/rho;
        vk  = sqrt(vk);

        for (int j = 0; j<jmax; j++)
        {
          udr[ib].r[j] = rho;
          udr[ib].p[j] = P;
          udr[ib].u[j] = get_viscous_vr(r, z);
          udr[ib].v[j] = vk;
          udr[ib].w[j] = 0.0;
          udr[ib].y[j] = zya[j];
          udr[ib].dy[j] = zdy[j];
          udr[ib].yc[j] = zyc[j];
          udr[ib].yvol[j] = get_vol( zya[j] , zdy[j] , 1 );
        }

        #if FARGO_flag == 1
        udr[ib].rot_v = vk - 1.0*udr[ib].xc;
        #elif FARGO_flag == 2
        udr[ib].rot_v = (1.0 - 1.0)*udr[ib].xc;
        #else
        udr[ib].rot_v = 0.0;
        #endif
        udr[ib].rot_j = 0;
      }
    }
  }

  if (ntop==3)
  {
    for (int k = 0; k<n_pad; k++)
    {
      for (int i = 0; i<imax; i++)
      {
        ib = k + n_pad*i;
        top[ib].z = zza[kmax-1] + zdz[kmax-1]*(double)(k+1);
        top[ib].dz = zdz[kmax-1];
        top[ib].zc = top[ib].z + 0.5*top[ib].dz;
        top[ib].zvol = get_vol( top[ib].z , top[ib].dz , 2 );

        top[ib].x = zxa[i];
        top[ib].dx = zdx[i];
        top[ib].xc = zxc[i];
        top[ib].xvol = get_vol( zxa[i] , zdx[i] , 0 );

        rr  = zxc[i];

        if (ngeomz==5)
        { 
          z = rr*cos(top[ib].zc);
          r = sqrt(rr*rr - z*z);
          dP = get_dP_s(rr, top[ib].zc);
        }
        else if (ngeomz==0)
        {
          r = rr;
          z = top[ib].zc;
          dP = get_dP_dr(r,z);
        }

        rho = get_rho(r,z);
        P = get_P(r,z);
        vk  = 1.0/rr;
        vk += rr*dP/rho;
        vk  = sqrt(vk);

        for (int j = 0; j<jmax; j++)
        {
          top[ib].r[j] = rho;
          top[ib].p[j] = P;
          top[ib].u[j] = get_viscous_vr(r, z);
          top[ib].v[j] = vk;
          top[ib].w[j] = 0.0;
          top[ib].y[j] = zya[j];
          top[ib].dy[j] = zdy[j];
          top[ib].yc[j] = zyc[j];
          top[ib].yvol[j] = get_vol( zya[j] , zdy[j] , 1 );
        }

        #if FARGO_flag == 1
        top[ib].rot_v = vk - 1.0*top[ib].xc;
        #elif FARGO_flag == 2
        top[ib].rot_v = (1.0 - 1.0)*top[ib].xc;
        #else
        top[ib].rot_v = 0.0;
        #endif
        top[ib].rot_j = 0;
      }
    } 
  }

  #endif
  return;
}

