void open_output_files(ofstream &wfile, string sfname)
{
   wfile.open(sfname.c_str(), ios::out);
   wfile.precision(16);

  return;
}

void open_binary_file(ofstream &wfile, string sfname)
{
  wfile.open(sfname.c_str(), ios::out | ios::binary);
  return;
}

void open_binary_file(ifstream &wfile, string sfname)
{
  wfile.open(sfname.c_str(), ios::in | ios::binary);
  return;
}

void copy_bfile(ifstream &bfile, GPU_plan *set)
{
  for (int n=0; n<nDev; n++)
    bfile.read((char*)set[n].h_rings, set[n].memsize);
  return;
}

void close_output_files(ofstream& wfile)
{
  wfile.close();
  
  return;
}

void write_har_file(double m, string output)
{
  ofstream plot;
  open_output_files(plot, output);

  double an[imax];
  double bn[imax];
  for (int i=0; i<imax; i++)
  {
    an[i]=0.0;
    bn[i]=0.0;
    for (int j=0; j<jmax; j++)
    {
      an[i] += sin(m*zyc[j])*zro[i][j][0];
      bn[i] += cos(m*zyc[j])*zro[i][j][0];
    }
    plot << zxc[i] << " " << an[i] << " " << bn[i] << endl;
    cout << zxc[i] << " " << an[i] << " " << bn[i] << endl;
  }
  return;
}

void write_nonlinear_file(string output)
{
  ofstream plot;
  open_output_files(plot, output);

  for (int i=0; i<imax; i++)
  {
    for (int j=0; j<jmax; j++)
    {
      plot << zxc[i] << " " << zyc[j] << " " << zro[i][j][0] << endl;
    }
  }
  return;
}

double grid_den(double x, int i, int j, double* xf)
{
  sdp x2, x3, h2, h3, A, B, C;
  x -= xf[i-1];

  C = zro[i-1][j][0];

  x2 = xf[i]  -xf[i-1];
  x3 = xf[i+1]-xf[i-1];

  h2 = (zro[i][j][0] - C)/x2;
  h3 = (zro[i+1][j][0] - C)/x3;

  A = (h3-h2)/(x3-x2);
  B = h2 - A*x2;
  
  return A*x*x + B*x + C;
}

double grid_ux(double x, int i, int j, double* xf)
{
  sdp x2, x3, h2, h3, A, B, C;
  x -= xf[i-1];

  C = zux[i-1][j][0];

  x2 = xf[i]  -xf[i-1];
  x3 = xf[i+1]-xf[i-1];

  h2 = (zux[i][j][0] - C)/x2;
  h3 = (zux[i+1][j][0] - C)/x3;

  A = (h3-h2)/(x3-x2);
  B = h2 - A*x2;
  
  return A*x*x + B*x + C;
}

void write_output_file(ofstream &ofile)
{
  double rho1, rho2;
  for (int i=0; i<imax; i++)
  {
    rho1 = 0.0;
    rho2 = 0.0;
    for (int j=0; j<jmax; j++)
    {
       rho1 += zro[i][j][kmax-1]*zdy[j];
       rho2 += get_surface_density(i, j)*zdy[j];
    }
    rho1 /= twopi;
    rho2 /= twopi;
    rho2 /= 0.030634144;
    ofile << zxc[i] << " " << rho1 << " " << rho2 << endl;
  }
/*
  double R1 = 1.0 - 0.03;//0.02236067977;
  double R2 = 1.0 + 0.03;//0.02236067977;
  double z;
  double x1[3], x2[3];
  double rho1, rho2;
  for (int k=0; k<kmax; k++)
  {
    z = 0.057*((double)k/(double)kmax);
    x1[0] = sqrt(R1*R1 + z*z);
    x2[0] = sqrt(R2*R2 + z*z);
    x1[2] = atan(R1/z);
    x2[2] = atan(R2/z);
    rho1 = 0.0;
    rho2 = 0.0;
    for (int j=0; j<jmax; j++)
    {
       x1[1] = zyc[j];
       x2[1] = zyc[j];
       rho1 += get_rho_sim(x1)*zdy[j];
       rho2 += get_rho_sim(x2)*zdy[j];
    }
    rho1 /= twopi;
    rho2 /= twopi;
    if (rho2!=0.0) ofile << z << " " << rho1 << " " << rho2 << " " << rho1/rho2 << endl;
    else ofile << z << " " << rho1 << " " << rho2 << " " << 0.0 << endl;
  }
  return;
*/
}

void write_rates_file(ofstream &ofile, double *grate[10])
{
  int mmax = 20;
  double avg;
  double var;
  for (int m=0; m<mmax; m++)
  {
    avg = 0.0;
    var = 0.0;
    ofile << " " << m+1;

    for (int i=0; i<10; i++)
    {
      avg += grate[i][m];
    }
    avg /= 10.0;
    ofile << " " << avg;

    for (int i=0; i<10; i++)
    {
      var += pow( grate[i][m]-avg , 2 );
    }
    var /= 9.0;
    ofile << " " << sqrt(var);
    ofile << endl;;
  }

  return;
}

double m_dot_p()
{
  double r_H = max(0.05, pow(0.001/3.0,1.0/3.0));

  double tmp1 = 0.0;
  double tmp2 = 0.0;

  for (int i=0; i<imax; i++)
  {
    if (abs(zxc[i]-1.0)/r_H<2.0)
    {
      for (int j=0; j<jmax; j++)
      { 
        if (abs(zyc[j]-pi)/r_H<2.0)
        {
          tmp1 += zro[i][j][0]*zxc[i]*zux[i][j][0]*zdy[j]*zdx[i];
        }
      }
      tmp2 += zdx[i];
    }
  }

  return tmp1/tmp2/(3.0*pi*0.001*0.05*0.05);
}

double m_dot_g()
{
  double r_H = max(0.05, pow(0.001/3.0,1.0/3.0));

  double tmp1 = 0.0;
  double tmp2 = 0.0;

  for (int i=0; i<imax; i++)
  {
    if (abs(zxc[i]-1.0)/r_H<2.0)
    {
      for (int j=0; j<jmax; j++)
      { 
        if (abs(zyc[j]-pi)/r_H>2.0)
        {
          tmp1 += zro[i][j][0]*zxc[i]*zux[i][j][0]*zdy[j]*zdx[i];
        }
      }
      tmp2 += zdx[i];
    }
  }

  return tmp1/tmp2/(3.0*pi*0.001*0.05*0.05);
}
