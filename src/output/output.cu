#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "variable_types.h"
#include "global.h"
#include "ppm.h"
#include "output.h"
#include "cuSafe.cu"

#include "device_func.cu"

void open_output_file(ofstream &wfile, string sfname)
{
  ios::openmode wrmode;
  wrmode=ios::out;

  wfile.open(sfname.c_str(), wrmode);
  wfile.precision(16);

  return;
}

void append_output_file(ofstream &wfile, string sfname)
{
  ios::openmode wrmode;
  wrmode=ios::app;

  wfile.open(sfname.c_str(), wrmode);
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

void save_cells(ofstream &ofile, GPU_plan *set)
{
  for (int n=0; n<nDev; n++)
  {
    ofile.write((char*)set[n].h_rings, set[n].memsize);
  }
  return;
}

string create_label()
{
  string label;
  label = int_to_string(imax)+"x"+int_to_string(jmax)+"x"+int_to_string(kmax)+"_c"+int_to_string(round(sc_h/0.001));
  #if opac_flag == 1
  label += "_b"+int_to_string(B0/0.001)+"_w"+int_to_string(buff/0.001);
  #endif
  #if plnt_flag == 2
  label += "_pm"+int_to_string(round(M_p/0.000003))+"_s"+int_to_string(round(100.0*rs_fac));
  #elif plnt_flag == 1
  label += "_p"+int_to_string(round(M_p/0.000003))+"_s"+int_to_string(round(100.0*rs_fac));
  #endif
  label += "_i"+int_to_string(xmin*100.0)+"_o"+int_to_string(xmax*100.0);
  //if (visc_flag==1) label += "_a"+int_to_string(round((vis_nu/sc_h/sc_h)*100000.0));
  if (visc_flag==1) label += "_a"+int_to_string(round(ss_alpha*10000.0));
  #if EOS == 1
  label += "_g" + int_to_string(round(gam*10.0));
  #elif EOS == 0
  if (p_alpha + 0.5*p_beta - 1.5 == 1.5) label += "_flatvor";
  if (p_beta == 0.0) label += "_iso";
  else label += "_b"+int_to_string(p_beta*10.0);
  #elif EOS == 2
  label += "_ad";
  #endif
  #if FrRot_flag == 1
  label += "_rot";
  #endif
  #if FARGO_flag == 1
  label += "_fargo";
  #endif
  return label;
}

void write_para_file(ofstream &ofile, string &mainp, string &label)
{
  ofile << imax << endl;
  ofile << jmax << endl;
  ofile << kmax << endl;
  ofile << nDev << endl;
  ofile << gam  << endl;
  ofile << sc_h << endl;
  ofile << tmovie << endl;
  ofile << mainp << endl;
  ofile << label << endl;
  ofile << grspx << endl;
  ofile << grspy << endl;
  ofile << grspz << endl;
  ofile << xmin << " " << xmax << endl;
  ofile << ymin << " " << ymax << endl;
  ofile << zmin << " " << zmax << endl;

  return;
}

void write_initial_file(ofstream &ofile, sdp* zxa, sdp* zdx)
{
  for (int i=0; i<imax; i++)
  {
    ofile << zxa[i];
    ofile << " " << zdx[i];
    ofile << endl;
  }
  return;
}
