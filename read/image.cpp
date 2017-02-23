float interpolate_tau(double rr, int index, sdp *tau)
{
  double A = (rr - zxa[index])/zdx[index];
  return (tau[index]*(1.0-A)) + (tau[index+1]*A);
}

//==============================================================================

void mode_image(string sfname, mglData &dat, mglData x, mglData y)
{
  mglGraph gr;

  string tword  = frame_num(round(1.0*simtime/twopi));
  //tword.insert(3, ".");
  tword = "t = "+tword+" orbits";
  char *tchar = new char[tword.size()+1];
  strcpy(tchar, tword.c_str());

  int i, j, k;

  //double r_hill = max(0.05,pow(0.01/3.0,1.0/3.0));

  double img_max = 1.5;
  double img_min = 0.0;
  double phi_max = pi+pi;
  double phi_min =-pi+pi;

  gr.SetSize(768,800);
  gr.SetFontSizePT(6);
  gr.SetPlotFactor(1.2);
  gr.Title(tchar);

  //---------------------------------------------------------------------
  double tmp;
  int ind;
  //---------------------------------------------------------------------

  gr.SetFontSizePT(8);

  float C_min = 0.0;
  float C_max = 2.0;

  #ifndef xy_flag
  gr.SetTicks('x', (img_max-img_min)/5.0, 0, 1.0);
  gr.SetTicks('y', (phi_max-phi_min)/5.0, 0, 0.0);
  gr.SetRanges(img_min,img_max,phi_min-pi,phi_max-pi,C_min,C_max);
  #else
  gr.SetTicks('x', 0.5, 4, 0.0);
  gr.SetTicks('y', 0.5, 4, 0.0);
  gr.SetRanges(-img_max,img_max,-img_max,img_max,C_min,C_max);
  #endif

  gr.SetTicks('c', -6, 0);
  gr.Colorbar("kubnclyqr");//"gcbBkRrqy");

  double fix = 2.0*(twopi-zyc[jmax-1]);
  //double norm = 0.8125*sqrt(pi/2.0)*0.03;
  double norm = sqrt(pi/2.0)*0.03;//(5.0*pi/32.0)*sc_h*sqrt(2.0/0.4);//
  double M_th = 0.57;//0.2*0.000015/pow(0.03,3);
  //double M_th = 0.1/pow(0.12,3);
  double amp = min( pow(1.0+3.0*M_th, 1.0/3.0)-1.0, 1.5);

  for (i=0; i<imax; i++)
  { 
    for (j=0; j<jmax; j++)
    {
      ind = i+imax*j;
      tmp = get_surface_density(i,j);
      dat.a[ind]= tmp/pow(x.a[ind],-1.5);//(tmp/norm/pow(x.a[ind],-1.5) - 1.0)/amp;//pow(zxc[i],-1.0);//
      //if (j==0) cout << dat.a[ind] << endl;
      //dat.a[ind] = log10(tmp*pow(x.a[ind],1.0)/norm);
      #ifdef xy_flag
      tmp      = x.a[ind]*cos(y.a[ind]+(fix*(double)j/(double)jmax));
      y.a[ind] = x.a[ind]*sin(y.a[ind]+(fix*(double)j/(double)jmax));
      x.a[ind] = tmp;
      #endif
    }
  }

  gr.Box();
  gr.Axis("xy");
  #ifndef xy_flag
  gr.Label('y',"{\\phi} [ radian ]",0);
  gr.Label('x',"R",0);
  #else
  //gr.Label('y',"y",0);
  //gr.Label('x',"x",0);
  //gr.Circle(mglPoint(0.0,0.0),0.5,"#g;2");
  //gr.Mark(mglPoint(1.0,0.0),".m5");  
  //gr.Circle(mglPoint(0.0,0.0),0.3,"#g;1");
  #endif

  gr.Dens(x,y,dat,"kubnclyqr");//"gcbBkRrqy"); 

  //---------------------------------------------------------------------

  sfname = sfname+".png";

  char *fname = new char[sfname.size()+1];
  strcpy(fname, sfname.c_str());

  cout << fname << " is saved." << endl;
  gr.WritePNG(fname);


  delete[] fname;
  //delete[] tchar;
  return;
}

//==============================================================================
double vz_range = 0.01;

void azi_image(string sfname, mglData &dat, mglData x, mglData y, int j)
{
  mglGraph gr;
  string tword  = frame_num(round(180.0*zyc[j]/pi));
  //tword.insert(3, ".");
  tword = "{\\phi} = "+tword+" degrees";
  char *tchar = new char[tword.size()+1];
  strcpy(tchar, tword.c_str());

  double r_H = pow(0.000015/3.0,1.0/3.0);
  double r_B = 0.000015/0.03/0.03;

  double img_max = 1.0;//1.5*r_B+1.0;
  double img_min = 0.125;//-1.5*r_B+1.0;
  double phi_max = 0.3;// 3.0*r_B;
  double phi_min = 0.0;// 0.0;

  gr.SetSize(768,768);
  gr.SetFontSizePT(6);
  //gr.Title(tchar);

  //---------------------------------------------------------------------
  double tmp;
  int ind;
  //---------------------------------------------------------------------

  gr.SetFontSizePT(8);

  float C_min = -6.0;//-vz_range;
  float C_max = 2.0;//vz_range;

  //gr.SetTicks('x', 0.5*r_hill, 0, 1.0);
  gr.SetTicks('x', (img_max-img_min)/5.0, 0, 1.0);
  gr.SetTicks('y', (phi_max-phi_min)/5.0, 0, 0.0);
  gr.SetRanges(img_min,img_max,phi_min,phi_max,C_min,C_max);

  gr.SetTicks('c', -5, 0);
  gr.Colorbar("kUbcgyqr_");
  //gr.Puts(mglPoint(1.0f,-4.81f,1),"{\\Sigma-\\Sigma_0}");

  double fix = 2.0*(twopi-zyc[jmax-1]);
  double norm = sc_h*sqrt(twopi)/2.0;//(5.0*pi/32.0)*sc_h*sqrt(2.0/0.4);//
  double maxv = 0.0;

  for (int i=0; i<imax; i++)
  { 
    for (int k=0; k<kmax; k++)
    {
      ind = i+imax*k;
      tmp = zro[i][j][k];//get_surface_density(i,j);
//      dat.a[ind]= -sin(zzc[k])*zuz[i][j][k] + cos(zzc[k])*zux[i][j][k];
      dat.a[ind] = log10(tmp);
      if (pow(x.a[ind]-1.0,2)+pow(y.a[ind],2)<r_B*r_B && dat.a[ind]<maxv) maxv = dat.a[ind];
    }
  }
  cout << maxv << endl;

  gr.Box();
  gr.Axis("xy");
  gr.Label('y',"z",0);
  gr.Label('x',"R",0);

  gr.Dens(x,y,dat,"kUbcgyr");
  //gr.Circle(mglPoint(1.0,0.0),r_B,"#k;");
  //---------------------------------------------------------------------

  sfname = sfname+"_j"+frame_num(j)+".png";

  char *fname = new char[sfname.size()+1];
  strcpy(fname, sfname.c_str());

  cout << fname << " is saved." << endl;
  gr.WritePNG(fname);


  delete[] fname;
  delete[] tchar;
  return;
}

void rad_image(string sfname, mglData &dat, mglData x, mglData y, int i)
{
  mglGraph gr;
  //string tword  = frame_num(round(180.0*zyc[j]/pi));
  //tword.insert(3, ".");
  //tword = "{\\phi} = "+tword+" degrees";
  //char *tchar = new char[tword.size()+1];
  //strcpy(tchar, tword.c_str());

  double r_H = pow(0.000015/3.0,1.0/3.0);
  double r_B = 0.000015/0.03/0.03;

  double img_max = 1.5*r_B+pi;
  double img_min =-1.5*r_B+pi;
  double phi_max = 3.0*r_B;
  double phi_min = 0.0;

  gr.SetSize(768,768);
  gr.SetFontSizePT(6);
  //gr.Title(tchar);

  //---------------------------------------------------------------------
  double tmp;
  int ind;
  //---------------------------------------------------------------------

  gr.SetFontSizePT(8);

  float C_min = -vz_range;
  float C_max = vz_range;

  //gr.SetTicks('x', 0.5*r_hill, 0, 1.0);
  gr.SetTicks('x', (img_max-img_min)/5.0, 0, pi);
  gr.SetTicks('y', (phi_max-phi_min)/5.0, 0, 0.0);
  gr.SetRanges(img_min,img_max,phi_min,phi_max,C_min,C_max);

  gr.SetTicks('c', -5, 0);
  gr.Colorbar("UbcgwyqrM_");
  //gr.Puts(mglPoint(1.0f,-4.81f,1),"{\\Sigma-\\Sigma_0}");

  double norm = sc_h*sqrt(twopi)/2.0;//(5.0*pi/32.0)*sc_h*sqrt(2.0/0.4);//
  double maxv = 0.0;

  for (int j=0; j<jmax; j++)
  { 
    for (int k=0; k<kmax; k++)
    {
      ind = j+jmax*k;
//      tmp = get_surface_density(i,j);
      dat.a[ind]= -sin(zzc[k])*zuz[i][j][k] + cos(zzc[k])*zux[i][j][k];
      if (pow(x.a[ind]-pi,2)+pow(y.a[ind],2)<r_B*r_B && -dat.a[ind]>maxv) maxv = dat.a[ind];
    }
  }
  cout << maxv << endl;

  gr.Box();
  gr.Axis("xy");
  gr.Label('y',"z",0);
  gr.Label('x',"{\\phi}",0);

  gr.Dens(x,y,dat,"UbcgwyqrM");
  gr.Circle(mglPoint(pi,0.0),r_B,"#k;");
  //---------------------------------------------------------------------

  sfname = sfname+"_i"+frame_num(i)+".png";

  char *fname = new char[sfname.size()+1];
  strcpy(fname, sfname.c_str());

  cout << fname << " is saved." << endl;
  gr.WritePNG(fname);


  delete[] fname;
  //delete[] tchar;
  return;
}

void mid_image(string sfname, mglData &dat, mglData x, mglData y)
{
  mglGraph gr;
/*
  string tword  = frame_num(round(100.0*simtime/twopi));
  tword.insert(3, ".");
  tword = "t = "+tword+" orbits";
  char *tchar = new char[tword.size()+1];
  strcpy(tchar, tword.c_str());
*/
  int i, j, k;
  k = 51;

  double r_H = pow(0.000015/3.0,1.0/3.0);
  double r_B = 0.000015/0.03/0.03;

  double img_max = 2.0*r_B+1.0;
  double img_min =-2.0*r_B+1.0;
  double phi_max = 2.0*r_B;
  double phi_min =-2.0*r_B;

  gr.SetSize(768,768);
  gr.SetFontSizePT(6);
  gr.SetPlotFactor(1.5);
  //gr.Title(tchar);

  //---------------------------------------------------------------------
  double tmp;
  int ind;
  //---------------------------------------------------------------------

  gr.SetFontSizePT(8);

  float C_min =-vz_range;
  float C_max = vz_range;

  gr.SetTicks('x', (img_max-img_min)/5.0, 0, 1.0);
  gr.SetTicks('y', (phi_max-phi_min)/5.0, 0, 0.0);//0.5*r_hill
  gr.SetRanges(img_min,img_max,phi_min,phi_max,C_min,C_max);

  gr.SetTicks('c', -5, 0);
  //gr.Colorbar("kRqyw");
  //gr.Colorbar("kubnclyqrm");
  gr.Colorbar("UbcgwyqrM_");

  //double fix = 2.0*(twopi-zyc[jmax-1]);
  //double norm = sc_h*sqrt(twopi)/2.0;//(5.0*pi/32.0)*sc_h*sqrt(2.0/0.4);//
  //double M_th = 0.18/pow(sc_h,3);
  //double amp = pow(1.0+3.0*M_th, 1.0/3.0)-1.0;

  for (i=0; i<imax; i++)
  { 
    for (j=0; j<jmax; j++)
    {
      ind = i+imax*j;
      //tmp = get_surface_density(i,j);
      //dat.a[ind] = log10(tmp/norm);
      //dat.a[ind] = log10(zro[i][j][kmax-1]);
      dat.a[ind]= -sin(zzc[k])*zuz[i][j][k] + cos(zzc[k])*zux[i][j][k];
    }
  }

  gr.Box();
  gr.Axis("xy");
  gr.Label('y',"{\\phi} [ radian ]",0);
  gr.Label('x',"R",0);

  gr.Dens(x,y,dat,"UbcgwyqrM");
  gr.Circle(mglPoint(1.0,0.0),r_B,"#k;");
  //---------------------------------------------------------------------

  sfname = sfname+"_mid.png";

  char *fname = new char[sfname.size()+1];
  strcpy(fname, sfname.c_str());

  cout << fname << " is saved." << endl;
  gr.WritePNG(fname);


  delete[] fname;
  //delete[] tchar;
  return;
}


void mode_image_rz(string sfname, mglData &dat, mglData &dat2, mglData &x, mglData &y)
{
  mglGraph gr;
  string tword  = frame_num(round(100.0*simtime/twopi));
  tword.insert(3, ".");
  tword = "t = "+tword+" orbits";
  char *tchar = new char[tword.size()+1];
  strcpy(tchar, tword.c_str());

  int i, j, k;

  double r_hill = max(0.05,pow(0.01/3.0,1.0/3.0));

  double img_max = 1.2;
  double img_min = 0.8;
  double phi_max = 0.2;
  double phi_min =-0.2;

  gr.SetSize(768,768);
  gr.SetFontSizePT(12);
  gr.Title(tchar);

  //---------------------------------------------------------------------
  double tmp;
  int ind;
  //---------------------------------------------------------------------

  //gr.SubPlot(2,1,1);
  gr.SetFontSizePT(11);

  gr.SetTicks('x', 0.05, 4, 1.0);
  gr.SetTicks('y', 0.05, 0, 0.0);
  gr.SetRanges(img_min,img_max,phi_min,phi_max);

  for (k=0; k<kmax; k++)
  { 
    for (i=0; i<imax; i++)
    {
      ind = i+imax*k;
      dat.a[ind]= log10(zro[i][jmax/2-1][k]);
    }
  }

  double C_min=-2.0;
  double C_max= 2.0;

  gr.SetTicks('c', -5, 0, 0.0);
  gr.Colorbar("mubgyrR");
  #ifndef xy_flag
  gr.Puts(mglPoint(1.0f,-4.81f,1),"{log_{10}}{\\Sigma}");
  //gr.Puts(mglPoint(1.0, 3.3), "PEnGUIn");
  #else
  gr.Puts(mglPoint(0.0,(-4.81f/twopi)*2.0*img_max,1),"{log_{10}}{\\Sigma}");
  //gr.Puts(mglPoint(0.0,(3.3/twopi)*2.0*img_max), "PEnGUIn");
  #endif

  gr.Box();
  gr.Axis("xy");

  gr.Label('y',"{\\phi} [ radian ]",0);
  gr.Label('x',"R",0);

  gr.Dens(x,y,dat,"mubgyrR");

  //---------------------------------------------------------------------

  sfname = sfname+".png";

  char *fname = new char[sfname.size()+1];
  strcpy(fname, sfname.c_str());

  cout << fname << " is saved." << endl;
  gr.WritePNG(fname);

  delete[] fname;
  //delete[] tchar;

  return;
}

//==============================================================================

void image3D(string sfname, mglData &sig, mglData &rz, mglData &sig_x, mglData &sig_y, mglData &sig_vx, mglData &sig_vy,
                                                       mglData &rz_x,  mglData &rz_y,  mglData &rz_vx,  mglData &rz_vy)
{
  mglGraph gr;
  string tword  = frame_num(round(100.0*simtime/twopi));
  tword.insert(3, ".");
  tword = "t = "+tword+" orbits";
  char *tchar = new char[tword.size()+1];
  strcpy(tchar, tword.c_str());

  int ind;

  double x_max = 1.3;
  double x_min = 0.7;
  double y_max = pi;
  double y_min =-pi;

  gr.SetSize(1536,768);
  gr.SetFontSizePT(14);
  gr.Title(tchar);

  double cmin= -1.0;
  double cmax=  1.0;

  gr.SetTicks('c', -5, 0);
  gr.Colorbar("kRroyw");
  gr.Puts(mglPoint(-1.2f, -1.4f, 0.f),"{log_{10}}");
  //gr.SetMeshNum(50);
  //---------------------------------------------------------------------

  gr.SubPlot(2,1,0);
  gr.SetFontSizePT(13);

  //gr.SetTicks('x', 0.05, 4, 1.0);
  //gr.SetTicks('y', 0.05, 4, 0.0);
  gr.SetTicks('x', 0.1, 3, 1.0);
  gr.SetTicks('y', 1.0, 3, 0.0);
  gr.SetRanges(x_min,x_max,y_min,y_max,cmin,cmax);

  for (int j=0; j<jmax; j++)
  { 
    for (int i=0; i<imax; i++)
    {
      ind = i+imax*j;
      if (sig_x.a[ind]<=x_max && sig_x.a[ind]>=x_min && sig_y.a[ind]<=y_max && sig_y.a[ind]>=y_min)
      {
        //sig.a[ind] = (zro[i][j][kmax/2]+zro[i][j][kmax/2-1])/2.0;
        //sig.a[ind] = get_surface_val(sig_x.a[ind], j);
        //sig_vx.a[ind] = get_surface_vx(sig_x.a[ind], j)/sig.a[ind];
        //sig_vy.a[ind] = get_surface_vy(sig_x.a[ind], j)/sig.a[ind];

        //sig.a[ind] /= 0.05*sqrt(twopi);
        //sig.a[ind] = log10(sig.a[ind]);
        //sig_vy.a[ind] -= 1.0;

        //cout << sig_vy.a[ind] << " " << pow(sig_x.a[ind],-0.5) << endl;
        //printf("(%f, %f) : %f\n", sig_x.a[ind], sig_y.a[ind], sig.a[ind]);
      }
      else
      {
        sig.a[ind] = 0.0;
        sig_vx.a[ind] = 0.0;
        sig_vy.a[ind] = 0.0;
      }
    }
  }

  gr.Box();
  gr.Axis("xy");

  gr.Label('y',"{\\phi} [ radian ]",0);
  gr.Label('x',"R",0);

  gr.Dens(sig_x,sig_y,sig,"kRroyw");
  //gr.Vect(sig_x,sig_y,sig_vx, sig_vy,"g");

  //---------------------------------------------------------------------

  x_max = 1.1;
  x_min = 0.9;
  y_max = 0.1;
  y_min =-0.1;

  gr.SubPlot(2,1,1);
  gr.SetFontSizePT(13);

  gr.SetTicks('x', 0.05, 4, 1.0);
  gr.SetTicks('y', 0.05, 4, 0.0);
  gr.SetRanges(x_min,x_max,y_min,y_max,cmin,cmax);
  double sum;

  for (int k=0; k<kmax; k++)
  { 
    for (int i=0; i<imax; i++)
    {
      //sum = 0.0;
      //for (int j=0; j<jmax; j++)
      //{
      //  sum += zro[i][j][k];
      //}
      //sum /= (double)jmax;
      ind = i+imax*k;
      rz.a[ind] = (zro[i][jmax/2-1][k]+zro[i][jmax/2][k])/2.0;

      rz.a[ind] = log10(rz.a[ind]);
      //rz_vx.a[ind] = zux[i][jmax/2-1][k];
      //rz_vy.a[ind] = zuz[i][jmax/2-1][k];
    }
  }
/*
  C_min=-2.0;
  C_max= 2.0;

  gr.SetTicks('c', (C_max-C_min)/5.f, 0, 0.0);
  gr.CAxis(C_min, C_max);
  gr.Colorbar("mubgyrR",3,0.10,0.035,0.8,0.72);
  gr.Puts(mglPoint(1.0f,-0.306f,1),"{log_{10}}{\\rho}");
*/
  gr.Box();
  gr.Axis("xy");

  gr.Label('y',"z",0);
  gr.Label('x',"R",0);

  gr.Dens(rz_x,rz_y,rz,"kRroyw");
  //gr.Vect(rz_x,rz_y,rz_vx, rz_vy,"g");

  //---------------------------------------------------------------------

  sfname = sfname+".png";

  char *fname = new char[sfname.size()+1];
  strcpy(fname, sfname.c_str());

  cout << fname << " is saved." << endl;
  gr.WritePNG(fname);

  delete[] fname;
  //delete[] tchar;

  return;
}

//==============================================================================

void harm_image(double m, string sfname, mglData &dat, mglData &dat2, mglData &x, mglData &y)
{
  mglGraph gr;
  string tword  = frame_num(round(simtime/twopi));
  //tword.insert(2, ".");
  tword = "t = "+tword+" orbits";
  char *tchar = new char[tword.size()+1];
  strcpy(tchar, tword.c_str());

  int i, j;

  double r_hill = max(0.05,pow(0.01/3.0,1.0/3.0));

  double img_max = 1.5;
  double img_min = 0.5;
  double phi_max = pi+pi;
  double phi_min = -pi+pi;

  gr.SetSize(768,768);
  gr.SetFontSizePT(12);
  //gr.Title(tchar);

  //---------------------------------------------------------------------
  double tmp;
  int ind;
  //---------------------------------------------------------------------

  //gr.SubPlot(2,1,1);
  gr.SetFontSizePT(11);

  #ifndef xy_flag
  gr.SetTicks('x', 0.5, 4, 1.0);
  gr.SetTicks('y', 1.0, 0, 0.0);
  gr.SetRanges(img_min,img_max,phi_min-pi,phi_max-pi,zmin,zmax);
  #else
  gr.SetTicks('x', 0.5, 0, 0.0);
  gr.SetTicks('y', 0.5, 0, 0.0);
  gr.SetRanges(-img_max,img_max,-img_max,img_max,zmin,zmax);
  #endif

  float C_min = 10.f;
  float C_max =-10.f;
  double fix = 2.0*(twopi-zyc[jmax-1]);

  double an[imax];
  double bn[imax];
  for (i=0; i<imax; i++)
  {
    an[i]=0.0;
    bn[i]=0.0;
    for (j=0; j<jmax; j++)
    {
      an[i] += sin(m*zyc[j])*zro[i][j][0];
      bn[i] += cos(m*zyc[j])*zro[i][j][0];
    }
  }

  double norm = 0.0;

  for (j=0; j<jmax; j++)
  { 
    for (i=0; i<imax; i++)
    {
      ind = i+imax*j;
      dat.a[ind] = an[i]*sin(m*zyc[j]) + bn[i]*cos(m*zyc[j]);
      if (dat.a[ind]>C_max) C_max = dat.a[ind];
      if (dat.a[ind]<C_min) C_min = dat.a[ind];
      #ifdef xy_flag
      tmp      = x.a[ind]*cos(y.a[ind]+(fix*(double)j/(double)jmax));
      y.a[ind] = x.a[ind]*sin(y.a[ind]+(fix*(double)j/(double)jmax));
      x.a[ind] = tmp;
      #endif
      if (fabs(dat.a[ind])>norm) norm = fabs(dat.a[ind]);
    }
  }

  for (j=0; j<jmax; j++)
  { 
    for (i=0; i<imax; i++)
    {
      ind = i+imax*j;
      dat.a[ind] /= norm;
    }
  }


  C_min=-1.0;
  C_max=1.0;

  gr.SetTicks('c', (C_max-C_min)/5.f, 0, 0.0);
  gr.Colorbar("BbgwyrR");
  #ifndef xy_flag
  gr.Puts(mglPoint(1.0f,-4.81f,1),"{log_{10}}{\\Sigma}");
  gr.Puts(mglPoint(1.0, 3.3), "PEnGUIn");
  #else
  gr.Puts(mglPoint(0.0,(-4.81f/twopi)*2.0*img_max,1),"{log_{10}}{\\Sigma}");
  gr.Puts(mglPoint(0.0,(3.3/twopi)*2.0*img_max), "PEnGUIn");
  #endif

  gr.Box();
  gr.Axis("xy");
  #ifndef xy_flag
  gr.Label('y',"{\\phi} [ radian ]",0);
  gr.Label('x',"R",0);
  #else
  gr.Label('y',"y",0);
  gr.Label('x',"x",0);
  #endif

  gr.Dens(x,y,dat,"BbgwyrR");

  #ifdef xy_flag
  fix = twopi/(double)(jmax-1);
  for (j=0; j<jmax; j++)
  { 
    for (i=0; i<imax; i++)
    {
      ind = i+imax*j;
      dat.a[ind]=C_min;
      tmp      = img_min*1.01*((double)i/(double)(imax-1))*cos(fix*(double)j);
      y.a[ind] = img_min*1.01*((double)i/(double)(imax-1))*sin(fix*(double)j);
      x.a[ind] = tmp;
      
    }
  }
  gr.Dens(x,y,dat,"BbgwyrR");
  #endif

  //---------------------------------------------------------------------

  sfname = sfname+".png";

  char *fname = new char[sfname.size()+1];
  strcpy(fname, sfname.c_str());

  cout << fname << " is saved." << endl;
  gr.WritePNG(fname);

  delete[] fname;
  //delete[] tchar;

  return;
}
