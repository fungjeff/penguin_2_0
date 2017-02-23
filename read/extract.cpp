double mean(double *sample, int size)
{
  double m = 0.0;
  for (int i=0; i<size; i++) m+= sample[i];
  return m/(double)size;
}

double stdev(double *sample, double mean, int size)
{
  double s = 0.0;
  for (int i=0; i<size; i++) s+= pow( sample[i]-mean , 2 );
  return sqrt(s/(double)(size-1));
}

int find_min_index(double *y, int len)
{
  int i;
  int index = 0;

  for (i=1; i<len; i++)
  {
    if (y[i]<y[index])
    {
      index = i;
    }
  }
  return index;
}

int find_max_index(double *y, int len)
{
  int i;
  int index = 0;

  for (i=1; i<len; i++)
  {
    if (y[i]>y[index])
    {
      index = i;
    }
  }
  return index;
}

int find_val_index(double val, vector<double> &y, int len)
{
  int i;
  int index = -1;

  for (i=1; i<len; i++)
  {
    if (y[i-1]<val && y[i]>val)
    {
      index = i;
    }
  }
  return index;
}

void regress (vector<double> &x, vector<double> &y, double &S, double &B, double &err,
              int start, int len)
{
  double S0 = (double)(len);
  double Sx = 0.0;
  double Sxx = 0.0;
  double Sy = 0.0;
  double Sxy = 0.0;
  double Syy = 0.0;

  for (int i=start; i<start+len; i++)
  {
    Sx += x[i];
    Sxx += x[i]*x[i];
    Sy += y[i];
    Sxy += x[i]*y[i];
    Syy += y[i]*y[i];
  }

  double Varx = S0*Sxx - Sx*Sx;
  double Vary = S0*Syy - Sy*Sy;

  S = ( S0*Sxy - Sx*Sy )/ Varx;
  B = ( Sy - S*Sx )/S0;

  err = ( Vary - S*S*Varx ) / S0 / (S0-2.0);
  err = sqrt(S0*err / Varx);

  return;
}

void write_growth_file(ofstream &growth, vector<double> &t, vector<vector<double> > &m)
{
  int len = t.size();

  double S, B, err;

  int i, j, ii;
  int tmp_i;
  const int nid = 4;
  const int num = nid*(nid+1)/2;

  int index[nid+1];
  double S_all[20][num];
  double B_all[20][num];

  double min_A = log(4.0e-7);
  double max_A = log(1.0e-4);
  double d_A = (max_A-min_A)/(double)nid;
  
  int fast_m = 0;
  int fast_t = len-1;

  for (j=1; j<20; j++)
  {
    if (m[j][fast_t] > m[fast_m][fast_t]) fast_m = j;
  }
  while (m[fast_m][fast_t]>max_A+log(10.0))
  {
    fast_t--;
    fast_m = 0;
    for (j=1; j<20; j++)
      if (m[j][fast_t] > m[fast_m][fast_t]) fast_m = j;
  }

  for (i=0; i<nid+1; i++) 
  {
    index[i] = len;
    tmp_i = find_val_index((double)(i*d_A)+min_A, m[fast_m], len);
    if (tmp_i != -1) if (tmp_i<index[i]) index[i] = tmp_i;
  }

  tmp_i = 0;
  for (i=0; i<nid; i++)
  for (ii=i+1; ii< nid+1; ii++)
  {
    for (j=0; j<20; j++)
    {
      regress(t, m[j], S, B, err, index[i], index[ii]-index[i]);
      S_all[j][tmp_i] = S;
      B_all[j][tmp_i] = B;
    }
    tmp_i++;
  }

  for (j=0; j<20; j++)
  {
    S = mean(&S_all[j][0], num);
    B = mean(&B_all[j][0], num);
    err = stdev(&S_all[j][0], S, num);
    growth << j+1 << " ";
    growth << S << " ";
    growth << B << " ";
    growth << err << endl;
  }
  
  return;
}
