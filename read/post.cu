#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <string>
#include <math.h>
#include <time.h>
#include </usr/include/mgl2/mgl.h>
#include "post_global.h"

#define img_flag 1
#define file_flag 0
#define bin_flag 1
#define xy_flag

const int imgDimx = 500;
const int imgDimy = 500;
const int imgDimz = 1;

vector<vector<vector<sdp> > > zro;
vector<vector<vector<sdp> > > zpr;
vector<vector<vector<sdp> > > zux;
vector<vector<vector<sdp> > > zuy;
vector<vector<vector<sdp> > > zuz;

int nDev;
sdp gam;
sdp sc_h;
sdp simtime;
sdp tmovie;
string mainp;
string label;
int grspx;
int grspy;
int grspz;
sdp xmin;
sdp xmax;
sdp ymin;
sdp ymax;
sdp zmin;
sdp zmax;

sdp *zxa;
sdp *zxc;
sdp *zdx;
sdp *zya;
sdp *zyc;
sdp *zdy;
sdp *zza;
sdp *zzc;
sdp *zdz;

//#include "../cuSafe.cu"
#include "rk_high.cpp"
#include "transform.cpp"
#include "p_init.cpp"
#include "extract.cpp"

#include "image.cpp"
#include "p_output.cpp"

using namespace std;

void read_para(string paraname)
{
  sdp tmp;

  ifstream para;
  para.open(paraname.c_str(), ios::in);

  para >> tmp;
  para >> tmp;
  para >> tmp;
  para >> nDev;
  para >> gam;
  para >> sc_h;
  para >> tmovie;
  para >> mainp;
  para >> label;
  para >> grspx;
  para >> grspy;
  para >> grspz;
  para >> xmin;
  para >> xmax;
  para >> ymin;
  para >> ymax;
  para >> zmin;
  para >> zmax;

  zxa=new sdp[imax];
  zxc=new sdp[imax];
  zdx=new sdp[imax];

  zya=new sdp[jmax];
  zyc=new sdp[jmax];
  zdy=new sdp[jmax];

  zza=new sdp[kmax];
  zzc=new sdp[kmax];
  zdz=new sdp[kmax];

  para.close();

  return;
}

int main(int narg, char *args[])
{
  int npic;
  bool allpic;

  if (narg == 1)
  {
    sc_h = 0.03;
    grspx = 6;
    grspy = 6;
    grspz = 7;

    xmin = 0.7;
    xmax = 1.3;
    ymin = 0.0;
    ymax = 2.0;
    zmin = 0.0;
    zmax = 0.15;

    zxa=new sdp[imax];
    zxc=new sdp[imax];
    zdx=new sdp[imax];

    zya=new sdp[jmax];
    zyc=new sdp[jmax];
    zdy=new sdp[jmax];

    zza=new sdp[kmax];
    zzc=new sdp[kmax];
    zdz=new sdp[kmax];

    init_grid();

    string resfname = "res_c30x.dat";
    ofstream resfile;

    open_output_files(resfile, resfname);

    for (int i=0; i<imax; i++) 
    {
      //printf(" (%f, %f, %f) \n", zxa[i], zxc[i], zdx[i]); wait_f_r();
      resfile << zxc[i] << " " << zdx[i] << endl;
    }
    resfile.close();

    resfname = "res_c30y.dat";
    open_output_files(resfile, resfname);
    for (int k=0; k<jmax; k++) 
    {
      resfile << zyc[k] << " " << zdy[k] << endl;
    }
    resfile.close();

    resfname = "res_c30z.dat";
    open_output_files(resfile, resfname);
    for (int k=0; k<kmax; k++) 
    {
      resfile << zzc[k] << " " << zdz[k] << endl;
    }
    resfile.close();
    //for (int j=0; j<jmax; j++) {printf(" (%f, %f, %f) \n", zya[j], zyc[j], zdy[j]); wait_f_r();}
    //for (int k=0; k<kmax; k++) {printf(" (%f, %f, %f) \n", zza[k], zzc[k], zdz[k]); wait_f_r();}
    return 0;
  }
  else if (narg==3)
  {
    read_para(string(args[1]));

    allpic = false;
    #if bin_flag == 1
    npic = atof(args[2]);
    if (npic==-1)
    {
      npic = 0;
      allpic = true;
    }
    #else
    npic = 10000;
    #endif
  }
  else if (narg==4)
  {
    read_para(string(args[1]));

    allpic = false;
    npic = atof(args[2]);
    allpic = true;
  }
  else
  {
    cout << endl << " Incorrect number of arguments." << endl;
    return 1;
  }

  string imgname, bfname, wfname, gfname;
  ofstream wfile, gfile;
  ifstream bfile;

  imgname = mainp+"/images/pic_"+label+"_";
  wfname  = mainp+"/files/zprofile_"+label+".dat";
  bfname  = mainp+"/binary/binary_"+label+"_";
  //bfname  = "/media/_storage/hydro_data/planet3D/binary_"+label+"_";

  simtime = (double)npic * tmovie;

  GPU_plan *set = new GPU_plan[nDev];
  
  for (int n=0; n<nDev; n++) set[n].id = startid+n;
  for (int n=0; n<nDev; n++) set[n].N_ring = imax/nDev;
  for (int n=0; n<imax%nDev; n++) set[nDev-1-n].N_ring++;
  for (int n=0; n<nDev; n++) set[n].N_ring *= kmax;

  //P2P_all_enable(set);

  for (int n=0; n<nDev; n++)
  { 
    set[n].memsize = set[n].N_ring*sizeof(hydr_ring);
    set[n].h_rings = new hydr_ring[set[n].N_ring];

    if (n==0) set[n].istart = 0;
    else      set[n].istart = set[n-1].istart + set[n-1].iblk;
    set[n].iblk = set[n].N_ring/kmax;
    set[n].kstart = 0;
    set[n].kblk = kmax;
  }

  set_size_for_data();
  open_binary_file(bfile, bfname+frame_num(0));
  copy_bfile(bfile, set);
  copy_grid(set);
  bfile.close();
  //for (int i=0; i<imax; i++) {printf(" (%f, %f, %f) \n", zxa[i], zxc[i], zdx[i]); wait_f_r();}
  //for (int j=0; j<jmax; j++) {printf(" (%f, %f, %f) \n", zya[j], zyc[j], zdy[j]); wait_f_r();}
  //for (int k=0; k<kmax; k++) {printf(" (%f, %f, %f) \n", zza[k], zzc[k], zdz[k]); wait_f_r();}

  mglData rz(imax, kmax);
  mglData rz_x(imax, kmax);
  mglData rz_y(imax, kmax);
  mglData rz_vx(imax, kmax);
  mglData rz_vy(imax, kmax);

  for (int i=0; i<imax; i++)
    for (int k=0; k<kmax; k++)
    {
      rz_x.a[i+imax*k] = zxc[i]*sin(zzc[k]);
      rz_y.a[i+imax*k] = zxc[i]*cos(zzc[k]);
      if (i==imax/2-1) cout << k << " " << rz_y.a[i+imax*k] << endl;
    }

  mglData pz(jmax, kmax);
  mglData pz_x(jmax, kmax);
  mglData pz_y(jmax, kmax);

  for (int j=0; j<jmax; j++)
    for (int k=0; k<kmax; k++)
    {
      pz_x.a[j+jmax*k] = zyc[j];
      pz_y.a[j+jmax*k] = cos(zzc[k]);
    }


  mglData sig(imax, jmax);
  mglData sig_x(imax, jmax);
  mglData sig_y(imax, jmax);
  mglData sig_vx(imax, jmax);
  mglData sig_vy(imax, jmax);

  for (int i=0; i<imax; i++)
    for (int j=0; j<jmax; j++)
    {
      sig_x.a[i+imax*j] = zxc[i];
      sig_y.a[i+imax*j] = zyc[j]-pi;
    }

  #if bin_flag == 1
  open_binary_file(bfile, bfname+frame_num(npic));
  #endif
  if(!bfile)
  {
    cout << endl << " Data file " << bfname+frame_num(npic) << " does not exist." << endl;
    return 1;
  }

  double tmp1=0.0;
  double tmp2=0.0;
  int count=0;

  if (allpic)
  {
    while(bfile)
    {
      copy_bfile(bfile, set);
      update_grid(set);
      #if img_flag == 1
      mode_image(imgname+frame_num(npic), sig, sig_x, sig_y);
      //xy_image(imgname+"xy_"+frame_num(npic), dat, dat2);
      #endif

      #if file_flag == 1
      open_output_files(wfile, wfname);
      tmp1 = get_torque()*sc_h*sc_h/(sqrt(pi/2.0)*sc_h)/M_p/M_p;
      wfile << simtime << " " << tmp1 << endl;
      cout  << simtime << " " << tmp1 << endl;
      #endif

      bfile.close();
      npic++;
      count++;
      simtime += tmovie;
      open_binary_file(bfile, bfname+frame_num(npic));
    }
    #if file_flag == 1
    wfile.close();
    #endif
  }
  else
  {
    #if bin_flag == 1
    copy_bfile(bfile, set);
    update_grid(set);
    #else
    ifstream para;
    para.open(string(args[2]).c_str(), ios::in);
    cout << imax << " " << jmax << " " << kmax << endl;
    double temp;
    for (int i=0; i<imax; i++)
      for (int j=0; j<jmax; j++)
        for (int k=0; k<kmax; k++)
        {
          para >> temp;
          para >> temp;
          para >> temp;

          para >> temp;
          zro[i][j][k] = temp;

          para >> temp;
          zux[i][j][k] = temp;

          para >> temp;
          zuy[i][j][k] = temp;

          para >> temp;
          zuz[i][j][k] = temp;
        }

    #endif
/*
    ios::openmode wrmode;
    wrmode=ios::out;

    ofstream wfile;
    string sfname = "z_acc"+label+".dat";

    wfile.open(sfname.c_str(), wrmode);
    wfile.precision(16);

    int ii = imax/2-1;
    int jj = jmax/2-1;
    double z, r, y;
    y = zyc[jj];
    double phi2, phi1;
    for (int k=1; k<kmax; k++)
    {
      z = cos(zzc[k])*zxc[ii];
      r = sin(zzc[k])*zxc[ii];
      phi2 = 1.0/sqrt(r*r+z*z) + 0.000015/sqrt((r-1.0)*(r-1.0)+(y-pi)*(y-pi)+z*z+0.00171*0.00171);
      z = cos(zzc[k-1])*zxc[ii];
      r = sin(zzc[k-1])*zxc[ii];
      phi1 = 1.0/sqrt(r*r+z*z) + 0.000015/sqrt((r-1.0)*(r-1.0)+(y-pi)*(y-pi)+z*z+0.00171*0.00171);
      wfile << cos(zza[k]) << " " << zro[ii][jj][k] << " " << zuz[ii][jj][k] << " " << -(zpr[ii][jj][k]-zpr[ii][jj][k-1])/(zzc[k]-zzc[k-1])/(0.5*(zro[ii][jj][k]+zro[ii][jj][k-1])) + (phi2-phi1)/(zzc[k]-zzc[k-1]) << " " << +(phi2-phi1)/(zzc[k]-zzc[k-1]);

      z = cos(zzc[k])*zxc[ii];
      r = 1.0;
      phi2 = 1.0/sqrt(r*r+z*z) + 0.000015/sqrt((r-1.0)*(r-1.0)+(y-pi)*(y-pi)+z*z+0.00171*0.00171);
      z = 0.066;
      r = 1.0;
      phi1 = 1.0/sqrt(r*r+z*z) + 0.000015/sqrt((r-1.0)*(r-1.0)+(y-pi)*(y-pi)+z*z+0.00171*0.00171);
      wfile << " " << pow( (0.4/0.03/0.03)*(phi2-phi1), 1.0/0.4 ) << endl;

    }
    cout << zxc[ii]-1.0 << " " << y-pi << endl;
    wfile.close();*/
    #if img_flag == 1
    for (int j=0; j<jmax; j++) cout << j << " " << zux[imax-1][j][kmax-1] << endl; 
    mode_image(imgname+frame_num(npic), sig, sig_x, sig_y);
    //mid_image(imgname+frame_num(npic), sig, sig_x, sig_y);
    //for (int j=0; j<jmax; j++)
      //azi_image(imgname+frame_num(npic), rz, rz_x, rz_y, j);
      //azi_image(imgname+frame_num(npic), rz, rz_x, rz_y, jmax/2-1-15);
      //azi_image(imgname+frame_num(npic), rz, rz_x, rz_y, jmax/2-1-10);
      //azi_image(imgname+frame_num(npic), rz, rz_x, rz_y, jmax/2-1-5);
      //azi_image(imgname+frame_num(npic), rz, rz_x, rz_y, jmax/2-1);
      //azi_image(imgname+frame_num(npic), rz, rz_x, rz_y, jmax/4-1);
      //azi_image(imgname+frame_num(npic), rz, rz_x, rz_y, jmax-1);
      //azi_image(imgname+frame_num(npic), rz, rz_x, rz_y, 3*jmax/4-1);
      //azi_image(imgname+frame_num(npic), rz, rz_x, rz_y, jmax/2-1+5);
      //azi_image(imgname+frame_num(npic), rz, rz_x, rz_y, jmax/2-1+10);
      //azi_image(imgname+frame_num(npic), rz, rz_x, rz_y, jmax/2-1+15);
      //azi_image(imgname+frame_num(npic), rz, rz_x, rz_y, jmax/2-1+5);
      //rad_image(imgname+frame_num(npic), pz, pz_x, pz_y, imax/2-1);
      //rad_image(imgname+frame_num(npic), pz, pz_x, pz_y, imax/2);
    //xy_image(imgname+"xy_"+frame_num(npic), dat, dat2);
    #endif
/*
    int count=0;
    for (int k=0; k<11; k++)
    {
    for (int j=0; j<jmax; j++)
    {
      for (int i=0; i<imax; i++)
      {
        if (zro[i][j][k]>1.0e-3) 
        {
          cout << k << " " << zyc[j] << " " << zxc[i] << " " << zro[i][j][k] << endl;
          count++;
        }
      }
    }
    cout << count << endl;
    wait_f_r();
    }
*/
    #if file_flag == 1
    //open_output_files(wfile, wfname);
    //write_output_file(wfile);
    //close_output_files(wfile);
    //ofstream wfile;
    //string sfname = "m_dot_"+label+".dat";
/*
    int ii = imax/2;

    double temp=0.0;
    double tmid=0.0;
    double m_dot = -1.5*0.001*sc_h*sc_h*twopi*sqrt(twopi)*sc_h;

    //wfile.open(sfname.c_str(), wrmode);
    //wfile.precision(16);

    for (int j=0; j<jmax; j++)
    {
      for (int k=0; k<kmax; k++)
      {
        temp += 0.5*zro[ii][j][k]   * zux[ii][j][k]   * zdz[k]*zxc[ii]   * zdy[j];
        temp += 0.5*zro[ii-1][j][k] * zux[ii-1][j][k] * zdz[k]*zxc[ii-1] * zdy[j];
      }
     
      tmid += 0.5*zro[ii][j][0]   * zux[ii][j][0]   * zxc[ii]   * zdy[j];
      tmid += 0.5*zro[ii-1][j][0] * zux[ii-1][j][0] * zxc[ii-1] * zdy[j];

      //wfile << (*a).y[j] << " " << temp << " " << tmid << endl;
      cout  << zyc[j] << " " << temp/m_dot << " " << tmid/(m_dot/sqrt(twopi)/sc_h) << endl;
    }
*/

    string resfname = "rho_216x432x72_q100_h120_fixedin_gap_veryhighvis_o40.dat";

    ofstream resfile;

    open_output_files(resfile, resfname);

    for (int i=0; i<imax; i++) 
    {
      for (int j=0; j<jmax; j++) 
      {
        for (int k=0; k<kmax; k++) 
        {
          resfile << i << " " << j << " " << k << " " << zro[i][j][k] << endl;
          //if (i==200 && j==jmax/2) cout << i << " " << j << " " << k << " " << zro[i][j][k] << endl;
        }
      }
    }
    resfile.close();


    resfname = "res_216x.dat";
    resfile;

    open_output_files(resfile, resfname);

    for (int i=0; i<imax; i++) 
    {
      resfile << i << " " << zxa[i] << " " << zdx[i] << endl;
    }
    resfile.close();

    resfname = "res_432y.dat";
    open_output_files(resfile, resfname);
    for (int j=0; j<jmax; j++) 
    {
      resfile << j << " " << zya[j] << " " << zdy[j] << endl;
    }
    resfile.close();

    resfname = "res_72z.dat";
    open_output_files(resfile, resfname);
    for (int k=0; k<kmax; k++) 
    {
      resfile << k << " " << zza[k] << " " << zdz[k] << endl;
    }
    resfile.close();

    #endif    

    count = 1;
  }

  //cout << "highest rho is: " << tmp1 << endl;

  cout << endl << " Data ends at n=" << npic << endl;

  return 0;
}
