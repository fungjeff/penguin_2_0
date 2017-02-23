sdp interpolate(vector< vector< vector< sdp > > > &vec, double r, double p, int index, int jndex)
{
  int iidex = index+1;
  int jjdex = jndex+1;
  if (jjdex==jmax) jjdex=0;
  double A = (r - zxa[index])/zdx[index];
  double B = (p - zya[jndex])/zdy[jndex];
  double B0 = (vec[index][jndex][0]*(1.0-A)) + (vec[iidex][jndex][0]*A);
  double B1 = (vec[index][jjdex][0]*(1.0-A)) + (vec[iidex][jjdex][0]*A);
  return (B0*(1.0-B)) + (B1*B);
}

double linear_interpolation(double x, double x1, double x2, double y1, double y2)
{
  double A = (x-x1)/(x2-x1);
  double B = 1.0-A;
  return A*y2 + B*y1;
}

double linear_interpolation_3D(double px, double py, double pz, double *x, double *y, double *z, double *v)
{
  double plane[4];
  int ind;

  for (int i=0; i<2; i++)
    for (int j=0; j<2; j++)
    {
      ind = j+2*i;
      plane[ind] = linear_interpolation(pz, z[0], z[1], v[0+2*ind], v[1+2*ind]);
    }

  double line[2];

  for (int i=0; i<2; i++)
  {
    line[i] = linear_interpolation(py, y[0], y[1], plane[0+2*i], plane[1+2*i]);
  }

  return linear_interpolation(px, x[0], x[1], line[0], line[1]);
}

double third_order_interpolation(double x, double x0, double x1, double x2, double x3, double y0, double y1, double y2, double y3)
{
  x -= x0;
  x1 -= x0;
  x2 -= x0;
  x3 -= x0;
  y1 = (y1-y0)/x1;
  y2 = (y2-y0)/x2;
  y3 = (y3-y0)/x3;

  double deno = (x2-x1)*(x3*x3-x1*x1) + (x1-x3)*(x2*x2-x1*x1);
  double C = y3*x1*x2*(x2-x1) + y2*x3*x1*(x1-x3) + y1*x2*x3*(x3-x2);
  double B = (x3*x3-x1*x1)*(y2-y1) - (x2*x2-x1*x1)*(y3-y1);
  double A = (x2-x1)*(y3-y1) + (x1-x3)*(y2-y1);

  C /= deno;
  B /= deno;
  A /= deno;

  return A*x*x*x + B*x*x + C*x + y0;
}

double third_order_interpolation_3D(double px, double py, double pz, double *x, double *y, double *z, double *v)
{
  double plane[16];
  int ind;

  for (int i=0; i<4; i++)
    for (int j=0; j<4; j++)
    {
      ind = j+4*i;
      plane[ind] = third_order_interpolation(pz, z[0], z[1], z[2], z[3], v[0+4*ind], v[1+4*ind], v[2+4*ind], v[3+4*ind]);
    }

  double line[4];

  for (int i=0; i<4; i++)
  {
    line[i] = third_order_interpolation(py, y[0], y[1], y[2], y[3], plane[0+4*i], plane[1+4*i], plane[2+4*i], plane[3+4*i]);
  }

  return third_order_interpolation(px, x[0], x[1], x[2], x[3], line[0], line[1], line[2], line[3]);
}

double get_rho_sim(double *x)
{
  bool rpass=false, ppass=false, zpass=false;

  int in[2], jn[2], kn[2];
  double gx[2], gy[2], gz[2], gr[8];

  for (int i=0; i<imax-1; i++)
  {
    if (x[0]>=zxc[i] && x[0]<zxc[i+1])
    {
      in[0] = i;
      in[1] = i+1;
      i=imax;
      rpass = true;
    } 
  }
  if (!rpass) return 0.0;

  for (int n=0; n<2; n++) 
  {
    gx[n] = zxc[in[n]];
  }

  for (int j=0; j<jmax-1; j++)
  {
    if (x[1]>=zyc[j] && x[1]<zyc[j+1])
    {
      jn[0] = j;
      jn[1] = j+1;
      j=jmax;
      ppass = true;
    }
  }

  for (int n=0; n<1; n++) 
  {
    if (ppass) gy[n] = zyc[jn[n]];
  }  

  if (!ppass)
  {
    jn[0] = jmax-1;
    jn[1] = 0;
    gy[0] = zyc[jmax-1];
    gy[1] = zyc[0]+twopi;
  }

  for (int k=0; k<kmax-1; k++)
  {
    if (x[2]>=zzc[k] && x[2]<zzc[k+1])
    {
      kn[0] = k;
      kn[1] = k+1;
      k=kmax;
      zpass = true;
    }
  }

  for (int n=0; n<2; n++) 
  {
    if (zpass) gz[n] = zzc[kn[n]];
  }

  if (x[2]>=zzc[kmax-1])
  {
    kn[0] = kmax-1;
    kn[1] = kmax-1;
    gz[0] = zzc[kmax-1];
    gz[1] = zzc[kmax-1]+zdz[kmax-1];
    zpass = true;
  }

  if (!zpass) return 0.0;

  for (int i=0; i<2; i++)
    for (int j=0; j<2; j++) 
      for (int k=0; k<2; k++)     
      {
        gr[k+2*(j+2*i)] = zro[in[i]][jn[j]][kn[k]];
      }

  return linear_interpolation_3D(x[0], x[1], x[2], gx, gy, gz, gr);
}

//########################## find Amplitude #############################

double find_amp(sdp r, sdp *amp)
{
  int index;
  for (int i=0; i<imax; i++)
  {
    if (zxa[i]>r) {index = i; i = imax;}
  }
  double Ai = (zxa[index]-r) / (zxa[index]-zxa[index-1]);
  double Aj = 1.0 - Ai;

  return (Aj*amp[index]) + (Ai*amp[index-1]);
}

double find_amp(sdp r, sdp2 *amp, int flag)
{
  int index = -1;
  for (int ii=0; ii<imax-1; ii++)
  {
    if(zxa[ii]<=r && zxa[ii+1]>r)
    {
      index=ii;
      ii=imax-1;
    }
  }

  if (index == -1) return 0.0;
  double Ai = (r-zxa[index]) / (zxa[index+1]-zxa[index]);
  double Aj = 1.0 - Ai;

  if (flag == 1) return (Aj*amp[index].y) + (Ai*amp[index+1].y);
  else           return (Aj*amp[index].x) + (Ai*amp[index+1].x);
}

//########################## Map to linear grid #############################

double get_surface_density(int i, int j)
{
  double sur = 0.0;

  if (ndim == 2)
  {
    return zro[i][j][0];
  }
  else if (ngeomz==0)
  {
    for (int k=0; k<kmax; k++)
    {
      sur += zro[i][j][k]*zdz[k];
    }
  }
  else if (ngeomz==5)
  {
    double x[3];
    double dz = zxc[i]*cos(zza[0])/(double)kmax;
    double z = 0.5*dz;

    x[1] = zyc[j];

    for (int k=0; k<kmax; k++)
    {
      x[0] = sqrt(zxc[i]*zxc[i]+z*z);
      x[2] = acos(z/x[0]);
      sur += get_rho_sim(x)*dz;
      //if (i==imax/2 && j == jmax/2) cout << x[2]-pi/2.0 << " " << get_rho_sim(x) << endl;
      z += dz;
    }
  }
  return sur;
}

double get_ave_v_phi(int i, int j)
{
  double ave = 0.0;
  double sur = 0.0;
  for (int k=0; k<kmax; k++)
  {
    ave += zuy[i][j][k]*zro[i][j][k]*zdz[k];
    sur += zro[i][j][k]*zdz[k];
  }
  return ave/sur;
}

//########################## Convert ###########################################

double convert(vector< vector< vector< sdp > > > &a, double img_max, double img_min, int ix, int iy, double phase=pi)
{
  while (phase>=twopi) phase-=twopi;

  double x = (((double)ix/(double)imgDimx)*2.0*img_max) - img_max;
  double y = (((double)iy/(double)imgDimy)*2.0*img_max) - img_max;
  double r = sqrt(x*x+y*y);
  double p;
  if (x>=0.0 && y>=0.0)
  {
    p = asin(y/r);
  }
  else if (x<=0.0 && y>=0.0)
  {
    p = pi - asin(y/r);
  }
  else if (x<=0.0 && y<=0.0)
  {
    p = pi - asin(y/r);
  }
  else
  {
    p = twopi + asin(y/r);
  }

  p -= phase;
  if (p<0.0) p+=twopi;

  int ir = floor((double)imgDimx*(r - img_min)/(img_max-img_min));
  int ip = floor((double)imgDimy*p/twopi);
  int irr = ir+1;
  int ipp = ip+1;

  if (ir<0 || irr>=imgDimx) return 0.0;

  if (ip >= imgDimy) {ip -= imgDimy; ipp -= imgDimy;}
  if (ip < 0) {ip += imgDimy; ipp += imgDimy;}
  if (ip == imgDimy-1) ipp = 0;

  double A = ((double)imgDimx*(r - img_min)/(img_max-img_min))-ir;
  double B = ((double)imgDimy*p/twopi)-ip;
  double B0 = (a[ir][ip][0]*(1.0-A)) + (a[irr][ip][0]*A);
  double B1 = (a[ir][ipp][0]*(1.0-A)) + (a[irr][ipp][0]*A);
  return (B0*(1.0-B)) + (B1*B);
}


double convert_vec(vector< vector< vector< sdp > > > &a, vector< vector< vector< sdp > > > &b, 
                   double img_max, double img_min, int ix, int iy , int dir)
{
  double x = (((double)ix/(double)imgDimx)*2.0*img_max) - img_max;
  double y = (((double)iy/(double)imgDimy)*2.0*img_max) - img_max;
  double r = sqrt(x*x+y*y);
  double p;
  if (x>=0.0 && y>=0.0)
  {
    p = asin(y/r);
  }
  else if (x<=0.0 && y>=0.0)
  {
    p = pi - asin(y/r);
  }
  else if (x<=0.0 && y<=0.0)
  {
    p = pi - asin(y/r);
  }
  else
  {
    p = twopi + asin(y/r);
  }

  int ir = floor((double)imgDimx*(r - img_min)/(img_max-img_min));
  int ip = floor((double)imgDimy*p/twopi);
  int irr = ir+1;
  int ipp = ip+1;

  if (ir<0 || irr>=imgDimx) return 0.0;

  if (ip >= imgDimy) {ip -= imgDimy; ipp -= imgDimy;}
  if (ip < 0) {ip += imgDimy; ipp += imgDimy;}
  if (ip == imgDimy-1) ipp = 0;

  double vr, vp;
  double A, B, B0, B1;
  A  = ((double)imgDimx*(r - img_min)/(img_max-img_min))-ir;
  B  = ((double)imgDimy*p/twopi)-ip;
  B0 = (a[ir][ip][0]*(1.0-A)) + (a[irr][ip][0]*A);
  B1 = (a[ir][ipp][0]*(1.0-A)) + (a[irr][ipp][0]*A);
  vr = (B0*(1.0-B)) + (B1*B);

  B0 = (b[ir][ip][0]*(1.0-A)) + (b[irr][ip][0]*A);
  B1 = (b[ir][ipp][0]*(1.0-A)) + (b[irr][ipp][0]*A);
  vp = (B0*(1.0-B)) + (B1*B);

  if (dir==0)
    return vr*cos(p)-r*vp*sin(p);
  else
    return vr*sin(p)+r*vp*cos(p);
}

//#############################################################################

double get_Mtol()
{
  double Mtol = 0;
  double vol;

  for (int i=0; i<imax; i++)
    for (int j=0; j<jmax; j++)
    {
      vol  = zdx[i]*zxc[i]*zdy[j];
      Mtol += zro[i][j][0]*vol;
    }
  return Mtol;
}

double potential(double r, double p)
{
  double cosfac = cos(p-pi);
  double M_s = 1.0-M_p;
  double a_p = 1.0;
  double a_s = a_p*(M_p/M_s);

  double r_p = sqrt(r*r + a_p*a_p - 2.0*a_p*r*cosfac + epsilon);
  double r_s = sqrt(r*r + a_s*a_s + 2.0*a_s*r*cosfac);

  return -M_s/r_s - M_p/r_p;
}

double get_torque()
{
  double stm = 1.0/(1.0+M_p);
  double plm = M_p*stm;
  double plx = 1.0*stm;
  double stx = 1.0*plm;

  double Rp, Rs, torque = 0.0;

  double cosfac, sinfac;

  double azi, rad, z;

  for (int j=0; j<jmax; j++)
  {
    azi = zyc[j];
    cosfac = cos(azi-pi);
    sinfac = sin(azi-pi);
    for (int i=0; i<imax; i++)
    {
      rad = zxc[i];
      for (int k=0; k<kmax; k++)
      {
        z = zzc[k];
        Rp = rad*rad + plx*plx - 2.0*plx*rad*cosfac + z*z;
        Rp = sqrt(Rp + 0.000003);

        Rs = sqrt(rad*rad + stx*stx + 2.0*stx*rad*cosfac + z*z);
        torque += zro[i][j][k]*(plm*plx*sinfac/(Rp*Rp*Rp) - stm*stx*sinfac/(Rs*Rs*Rs))*rad*zdy[j]*zdx[i]*zdz[k];
      }
    }
  }

  return torque;
}

double get_max_rho()
{
  double peak=0.0;
  for (int i=0; i<imax; i++)
    for (int j=0; j<jmax; j++)
      if (zro[i][j][0] > peak) peak = zro[i][j][0];
  return peak;
}

void get_velocity(double t, double *x, double *v)
{
  int in=0, jn=0, kn=0;
  int ip, jp, kp;
  for (int i=1; i<imax; i++) if (x[0]<zxc[i]) {in = i; i=imax;} 
  for (int j=1; j<jmax; j++) if (x[1]<zyc[j]) {jn = j; j=jmax;}
  for (int k=1; k<kmax; k++) if (x[2]<zzc[k]) {kn = k; k=kmax;}

  ip = in-1;
  if (jn==0) jp=jmax-1;
  else       jp = jn-1;
  kp = kn-1;

  double A = (x[0]-zxc[ip])/(zxc[in]-zxc[ip]);
  double B;
  if (jn==0) B = (x[1]-zyc[jp])/(twopi+zyc[jn]-zyc[jp]);
  else       B = (x[1]-zyc[jp])/(zyc[jn]-zyc[jp]);
  double C = (x[2]-zzc[kp])/(zzc[kn]-zzc[kp]);

  v[0] = A*B*C*zux[in][jn][kn] + (1.0-A)*B*C*zux[ip][jn][kn]
       + A*B*(1.0-C)*zux[in][jn][kp] + (1.0-A)*B*(1.0-C)*zux[ip][jn][kp]
       + A*(1.0-B)*C*zux[in][jp][kn] + (1.0-A)*(1.0-B)*C*zux[ip][jp][kn]
       + A*(1.0-B)*(1.0-C)*zux[in][jp][kp] + (1.0-A)*(1.0-B)*(1.0-C)*zux[ip][jp][kp];

  v[1] = A*B*C*zuy[in][jn][kn] + (1.0-A)*B*C*zuy[ip][jn][kn]
       + A*B*(1.0-C)*zuy[in][jn][kp] + (1.0-A)*B*(1.0-C)*zuy[ip][jn][kp]
       + A*(1.0-B)*C*zuy[in][jp][kn] + (1.0-A)*(1.0-B)*C*zuy[ip][jp][kn]
       + A*(1.0-B)*(1.0-C)*zuy[in][jp][kp] + (1.0-A)*(1.0-B)*(1.0-C)*zuy[ip][jp][kp];
  v[1] /= x[0]*sin(x[2]);
  v[1] -= 1.0;

  v[2] = A*B*C*zuz[in][jn][kn] + (1.0-A)*B*C*zuz[ip][jn][kn]
     + A*B*(1.0-C)*zuz[in][jn][kp] + (1.0-A)*B*(1.0-C)*zuz[ip][jn][kp]
     + A*(1.0-B)*C*zuz[in][jp][kn] + (1.0-A)*(1.0-B)*C*zuz[ip][jp][kn]
     + A*(1.0-B)*(1.0-C)*zuz[in][jp][kp] + (1.0-A)*(1.0-B)*(1.0-C)*zuz[ip][jp][kp];
  v[2] /= x[0];

  return;
}

void streamline(vector<double> &x_list, vector<double> &y_list, vector<double> &z_list)
{
  double dt;
  double x[3], tmpx[3], v[3], tmpv[3];

  x[0] = x_list[0];
  x[1] = y_list[0];
  x[2] = z_list[0];

  bool endpoint = false;
  double dtfac = 0.05;

  do
  {
    get_velocity(0.0, x, v);

    dt = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    dt = dtfac*0.05/dt;

    if (v[1]>0.0)
    {
      if (dt*v[1] >= twopi-x[1])
      {
        dt  *= (twopi-x[1])/(dt*v[1]);
      }
    }
    else
    {
      if (-dt*v[1] >= x[1])
      {
        dt  *= -x[1]/(dt*v[1]);
      }
    }

    for (int n = 0; n<3 ; n++) tmpx[n] = x[n] + dt*v[n];
    get_velocity(0.0, tmpx, tmpv);

    for (int n = 0; n<3 ; n++) v[n] = 0.5*(v[n] + tmpv[n]);

    dt = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    dt = dtfac*0.05/dt;

    if (v[1]>0.0)
    {
      if (dt*v[1] >= twopi-x[1])
      {
        dt  *= (twopi-x[1])/(dt*v[1]);
        endpoint = true;
      }
    }
    else
    {
      if (-dt*v[1] >= x[1])
      {
        dt  *= -x[1]/(dt*v[1]);
        endpoint = true;
      }
    }

    for (int n = 0; n<3 ; n++) x[n] = x[n] + dt*v[n];

    cout << x[0] << " " << x[1] << " " << x[2] << endl;
    cout << v[0] << " " << v[1] << " " << v[2] << endl << endl;
    wait_f_r();

    x_list.push_back(x[0]);
    y_list.push_back(x[1]);
    z_list.push_back(x[2]);

  } while (!endpoint);

  return;
}

void streamline_high(vector<double> &x_list, vector<double> &y_list, vector<double> &z_list)
{
  double t, dt;
  double x[3], v[3], err;

  x[0] = x_list[0];
  x[1] = y_list[0];
  x[2] = z_list[0];

  bool acc, endpoint = false;

  get_velocity(0.0, x, v);
  if (v[1]<0.0)
  {
    x[1] = pi+0.15; 
    get_velocity(0.0, x, v);
    y_list.clear(); 
  }

  dt = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
  dt = 0.05/dt;

  x_list.clear(); 
  y_list.clear(); 
  z_list.clear(); 

  x_list.push_back(x[0]*sin(x[2]));
  y_list.push_back(x[1]);
  z_list.push_back(x[0]*cos(x[2]));

  do
  {
    if (v[1]>0.0)
    {
      if (dt*v[1] >= pi+0.15-x[1])
      {
        dt  *= (pi+0.15-x[1])/(dt*v[1]);
        endpoint = true;
      }
    }
    else
    {
      if (-dt*v[1] >= x[1]-(pi-0.15))
      {
        dt  *= -(x[1]-(pi-0.15))/(dt*v[1]);
        endpoint = true;
      }
    }

    err = adap_rk853(t, x, dt, get_velocity, acc);
    get_velocity(t, x, v);

    if (!acc) endpoint = false;

    //cout << x[0]*sin(x[2]) << " " << x[1] << " " << x[0]*cos(x[2]) << endl;
    //cout << v[0] << " " << v[1] << " " << v[2] << endl << endl;
    //wait_f_r();

    x_list.push_back(x[0]*sin(x[2]));
    y_list.push_back(x[1]);
    z_list.push_back(x[0]*cos(x[2]));
    if (x_list.size()==1024) endpoint = true;

  } while (!endpoint);

  return;
}
