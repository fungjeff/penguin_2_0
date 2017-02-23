//#########################################################################################
double tan_eqn(double A, double D, double N)
{
  return sqrt(D/A) * tan( sqrt(A*D)*N );
}

double solve_tan(double x, double D, double N)
{
  double A = pow(pi/4.0/N,2)/D;
  double dx_dA;
  double diff = x-tan_eqn(A, D, N);

  do
  {
    dx_dA = (tan_eqn(A+0.0000001, D, N)-tan_eqn(A-0.0000001, D, N))/0.0000002;
    A += 0.1*diff/dx_dA;
    diff = x-tan_eqn(A, D, N);
  }while(fabs(diff/x)>1.e-14);
  
  return A;
}

double solve_grid_eqn(double a)
{
  double x, fx, fa, fxp, dx;

  fa = 1.0;

  x = (3.0-sqrt(24.0/a - 15.0))/2.0;
  fx = x/a + exp(-x);

  do
  {
    fxp = 1.0/a - exp(-x);

    dx = (fa-fx)/fxp;
    x += dx;
    fx = x/a + exp(-x);
  }while(fabs(fa-fx)>1.e-14);

  return x;
}

double solve_grid_eqn(double a, double b)
{
  double x, fx, fa, fxp, dx;

  fa = 1.0;

  x = 3.213842354076886751;
  fx = exp(x*(1.0 - b/(x+a))) - x/a;

  do
  {
    fxp = exp(x*(1.0 - b/(x+a)))*(1.0 - b/(x+a) + x*b/(x+a)/(x+a)) - 1.0/a;

    dx = (fa-fx)/fxp;
    x += dx;
    fx = exp(x*(1.0 - b/(x+a))) - x/a;
  }while(fabs(fa-fx)>1.e-14);

  return x;
}

//#########################################################################################

void grid(int start, int nzones, sdp xmin, sdp xmax, sdp* xa, sdp* xc, sdp* dx, int lg )
{
  if (lg==0)
  {
    double dxfac = (xmax - xmin) / (double)nzones;
    for (int n = start; n<start+nzones; n++)
    {
      xa[n] = xmin + ((double)(n-start))*dxfac;
      dx[n] = dxfac;
      xc[n] = xa[n] + 0.5*dx[n];
    }
    //for(int i=0; i<nzones; i+=4) cout << " " << i << " : " << xa[i]/twopi << " " << dx[i]/twopi << " " << (xa[i]+dx[i])/twopi << endl;
    //wait_f_r();
  }
  else if (lg==1)
  {
    double lg_dxfac = (log(xmax)-log(xmin))/(double)nzones;
    xa[start] = xmin;
    for (int n = start+1; n<start+nzones; n++)
    {
      xa[n] = xmin*exp( lg_dxfac*((double)(n-start)) );
      dx[n-1] = xa[n]-xa[n-1];
      xc[n-1] = xa[n-1] + 0.5*dx[n-1];
    }
    dx[start+nzones-1] = xmax - xa[start+nzones-1];
    xc[start+nzones-1] = xa[start+nzones-1] + 0.5*dx[start+nzones-1];

  }
  else if (lg==2)
  {
    double lg_dxfac = (log(xmax)-log(xmin))/(double)nzones;
    xa[start] = xmin;
    for (int n = start+1; n<start+nzones; n++)
    {
      xa[n] = xmin*exp( lg_dxfac*((double)(n-start)) );
      dx[n-1] = xa[n]-xa[n-1];
      xc[n-1] = xa[n-1] + 0.5*dx[n-1];
    }
    dx[start+nzones-1] = xmax - xa[start+nzones-1];
    xc[start+nzones-1] = xa[start+nzones-1] + 0.5*dx[start+nzones-1];
  }
  else if (lg==3)
  {
    double lg_dxfac = (log(xmax)-log(xmin))/(double)nzones;
    xa[start] = xmin;
    for (int n = start+1; n<start+nzones; n++)
    {
      xa[n] = xmin*exp( lg_dxfac*((double)(n-start)) );
      dx[n-1] = xa[n]-xa[n-1];
      xc[n-1] = xa[n-1] + 0.5*dx[n-1];
    }
    dx[start+nzones-1] = xmax - xa[start+nzones-1];
    xc[start+nzones-1] = xa[start+nzones-1] + 0.5*dx[start+nzones-1];
  }
  else if (lg==4)
  {
    double xmid = (xmax+xmin)/2.0;
    double xp = xmid - xmin;

    int N = nzones/2;
    xp -= N*dx_min;
    double power = log(1.0-(dx_max-dx_min)/xp)/log(1.0-1.0/N);

    double A = pow(xp,1.0/power)/(double)N;

    for (int i=0; i<N ;i++)
    {
      xa[i] = xmid - pow((double)(N-i)*A,power) - (double)(N-i)*dx_min;
    }

    for (int i=N; i<nzones ;i++)
    {
      xa[i] = pow((double)(i-N)*A,power) + (double)(i-N)*dx_min + xmid;
    }
    for(int i=0; i<nzones-1; i++)
    {
      dx[i] = xa[i+1]-xa[i];
      xc[i] = xa[i]+0.5*dx[i];
    }
    dx[nzones-1] = xmax-xa[nzones-1];
    xc[nzones-1] = xa[nzones-1]+0.5*dx[nzones-1];
  }
  else if (lg==5)
  {
    int low_N = ceil(0.0/dy_max);
    low_N += low_N%2;
    double low_size = low_N*dy_max;
    double xmid = (xmax+xmin)/2.0;
    double xp = xmid - xmin - low_size/2.0;

    int N = (nzones-low_N)/2;
    xp -= N*dy_min;
    double power = log(1.0-(dy_max-dy_min)/xp)/log(1.0-1.0/N);

    double A = pow(xp,1.0/power)/(double)N;

    int Nb1 = low_N/2;
    int Nb2 = low_N/2 + N;
    if (Nb2 != nzones/2) {cout << " zone 2 error in y grid. " << endl; wait_f_r();}
    int Nb3 = low_N/2 + 2*N;
    int Nb4 = low_N + 2*N;
    if (Nb4 != nzones) {cout << " zone 4 error in y grid. " << endl; wait_f_r();}

    for (int i=0; i<Nb1 ;i++)
    {
      xa[i] = dy_max*(double)i + xmin;
    }

    for (int i=Nb1; i<Nb2 ;i++)
    {
      xa[i] = xmid - pow((double)(Nb2-i)*A,power) - (double)(Nb2-i)*dy_min;
    }

    for (int i=Nb2; i<Nb3 ;i++)
    {
      xa[i] = pow((double)(i-Nb2)*A,power) + (double)(i-Nb2)*dy_min + xmid;
    }

    for (int i=Nb3; i<Nb4 ;i++)
    {
      xa[i] = dy_max*(double)((i+1)-Nb3) + xa[Nb3-1];
    }

    for(int i=0; i<nzones-1; i++)
    {
      dx[i] = xa[i+1]-xa[i];
      xc[i] = xa[i]+0.5*dx[i];
    }
    dx[nzones-1] = xmax-xa[nzones-1];
    xc[nzones-1] = xa[nzones-1]+0.5*dx[nzones-1];
  }
  else if (lg==6)
  {
    double xp = xmax - xmin;

    int N = nzones;
    xp -= N*dz_min;
    double power = log(1.0-(dz_max-dz_min)/xp)/log(1.0-1.0/N);

    double A = pow(xp,1.0/power)/(double)N;

    for (int i=0; i<nzones ;i++)
    {
      xa[i] = xmax - pow((double)(N-i)*A,power) - (double)(N-i)*dz_min;//pow((double)(i)*A,power) + (double)(i)*dz_min + xmin;
    }
    for(int i=0; i<nzones-1; i++)
    {
      dx[i] = xa[i+1]-xa[i];
      xc[i] = xa[i]+0.5*dx[i];
    }
    dx[nzones-1] = xmax-xa[nzones-1];
    xc[nzones-1] = xa[nzones-1]+0.5*dx[nzones-1];
  }
  return;
}

//#########################################################################################

__host__ __device__ sdp get_vol(sdp xa, sdp dx, int axis)
{
  sdp vol = 0.0;
  if (axis==0)
  {
    if (ngeomx==0)      vol = dx;
    else if (ngeomx==1) vol = dx*(xa + 0.5*dx);
    else if (ngeomx==2) vol = dx*(xa*(xa + dx) + dx*dx*third);
  }
  else if (axis==1)
  {
    if (ngeomy==0)      vol = dx;
    else if (ngeomy==3) vol = dx;
    else if (ngeomy==4) vol = dx;
  }
  else if (axis==2)
  {
    if (ndim==2)        vol = 1.0;
    else if (ngeomz==0) vol = dx;
    else if (ngeomz==5) vol = cos(xa)-cos(xa+dx); 
  }

  return vol;
}
