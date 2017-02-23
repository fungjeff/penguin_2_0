#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <math.h>
#include <time.h>

#include "variable_types.h"
#include "global.h"
#include "disk_profile.h"
#include "ppm.h"
#include "output.h"
#include "timestep.h"
#include "FARGO.h"
#include "cuSafe.cu"

using namespace std;

//=======================================================================
// CPU Grid
//=======================================================================

sdp *zxa;
sdp *zxc;
sdp *zdx;
sdp *zya;
sdp *zyc;
sdp *zdy;
sdp *zza;
sdp *zzc;
sdp *zdz;

#include "init.cpp"

void copy_bfile(ifstream &bfile, GPU_plan *set)
{
  for (int n=0; n<nDev; n++)
    bfile.read((char*)set[n].h_rings, set[n].memsize);
  return;
}

void show_init(sdp B0, double ttau)
{
  sdp *beta = new sdp[imax*jmax*kmax];
  SymDisk *val = new SymDisk[imax*kmax];

  zxa=new sdp[imax];
  zxc=new sdp[imax];
  zdx=new sdp[imax];

  zya=new sdp[jmax];
  zyc=new sdp[jmax];
  zdy=new sdp[jmax];

  zza=new sdp[kmax];
  zzc=new sdp[kmax];
  zdz=new sdp[kmax];

//-----------------------------------------------------------------------------------------
// Generate Grid
    init_grid();

//-----------------------------------------------------------------------------------------
// Initializing
  init_den(val);

  init_speed(val);

  double temp;
  int ind;
  double m_dot = -1.5*ss_alpha*sc_h*sc_h*twopi*sqrt(twopi)*sc_h;

  for (int i=0; i<imax; i++)
  {
    temp=0;
    for (int k=0; k<kmax; k++)
    {
      ind = i+imax*k;
      temp += val[ind].r*val[ind].u*twopi*zxc[i]*zdz[k];
    }
    cout << zxc[i] << " " << val[i+imax*9].r << " " << val[i].v << " " << val[i].u << " " << temp/m_dot << " " << val[ind].r*val[ind].u*twopi*zxc[i]/(-1.5*ss_alpha*sc_h*sc_h*twopi) << endl;
  }
  wait_f_r();
  for (int k=0; k<kmax; k++)
  {
     cout << k << " " << zza[k] << " " << zdz[k] << endl;
  }
  wait_f_r();
  for (int j=0; j<jmax; j++)
  {
     cout << j << " " << zya[j] << " " << zdy[j] << endl;
  }
  wait_f_r();
  for (int i=0; i<imax; i++)
  {
     cout << i << " " << zxa[i] << " " << zdx[i] << endl;
  }
/*
  for (int i=0; i<imax; i++)
  {
     cout << i << " " << zxa[i] << " " << zdx[i]/zxc[i] << " " << val[i+imax*(kmax-1)].u << endl;
  }
*/
  cout << (0.5*zxc[0]*zyc[0]/(val[imax*(kmax-1)].v+sqrt(get_cs2(zxc[0]*sin(zzc[kmax-1])))-zxc[0]))/(twopi*5.0) << endl;
  cout << val[imax*(kmax-1)].v << " " << sqrt(get_cs2(zxc[0]*sin(zzc[kmax-1]))) << " " << sqrt(val[imax*(kmax-1)].p/val[imax*(kmax-1)].r) << " " << zxc[0]*sin(zzc[kmax-1]) << endl;

  delete[] beta, val;
  return;
}

void kernel(bool restart, int startat, int morph, string morph_path, sdp B0, double ttau)
{
  string mainp = path_to_cwd();

  string label = create_label();
  string ifname, pfname, bfname, tfname;
  ofstream ifile, pfile, bfile, tfile;

  ifname  = mainp+"/files/initial_"+label+".dat";
  tfname  = mainp+"/files/time_"+label+".dat";
  pfname  = mainp+"/read/para_"+label+".dat";
  bfname  = mainp+"/binary/binary_"+label+"_";

  #if file_flag == 1
  ofstream result_file;
  string result_fname;
  result_fname  = mainp+"/files/result_"+label;
  #endif

  if (ndim==1 && jmax*kmax!=1){
    cout << "Requesting 1D problem but arrays are dimensioned for 2/3D" << endl;
    return;
  }
  if (ndim==2 && kmax!=1){
    cout << "Requesting 2D problem but arrays are dimensioned for 3D" << endl;
    return;
  }
  if (ndim==3 && kmax==1){
    cout << "Requesting 3D problem but arrays are dimensioned for 2D" << endl;
    return;
  }
//-----------------------------------------------------------------------------------------
// Begin by allocating device memory

  GPU_plan *set = new GPU_plan[nDev];

  //if (nDev==2 && startid==0) {set[0].id=0; set[1].id=2;}
  //else
  for (int n=0; n<nDev; n++) set[n].id = startid+n;
  for (int n=0; n<nDev; n++) set[n].N_ring = imax/nDev;
  for (int n=0; n<imax%nDev; n++) set[nDev-1-n].N_ring++;
  for (int n=0; n<nDev; n++) set[n].N_ring *= kmax;

  //P2P_all_enable(set);

  for (int n=0; n<nDev; n++)
  {
    CudaSafeCall( cudaSetDevice(set[n].id) );
    CudaSafeCall( cudaDeviceSetCacheConfig(cudaFuncCachePreferShared) );
    CudaSafeCall( cudaStreamCreate(&set[n].stream) );
    CudaSafeCall( cudaEventCreate(&set[n].event) );
   
    printf("Device %i contains %i rings.\n", set[n].id, set[n].N_ring);
    set[n].memsize = set[n].N_ring*sizeof(hydr_ring);

    if (n==0) set[n].istart = 0;
    else      set[n].istart = set[n-1].istart + set[n-1].iblk;
    set[n].iblk = set[n].N_ring/kmax;
    set[n].kstart = 0;
    set[n].kblk = kmax;
    
    set[n].sx_grid.x = jmax;
    set[n].sx_grid.y = set[n].kblk;
    set[n].sx_grid.z = 1;

    //printf(" %i, %i, %i \n", set[n].sx_grid.x, set[n].sx_grid.y, set[n].sx_grid.z );

    set[n].sy_grid.x = set[n].iblk;
    set[n].sy_grid.y = set[n].kblk;
    set[n].sy_grid.z = 1;

    //printf(" %i, %i, %i \n", set[n].sy_grid.x, set[n].sy_grid.y, set[n].sy_grid.z );

    set[n].sz_grid.x = set[n].iblk;
    set[n].sz_grid.y = jmax;
    set[n].sz_grid.z = (set[n].kblk/realarr) + (bool)(set[n].kblk%realarr);

    //printf(" %i, %i, %i \n", set[n].sz_grid.x, set[n].sz_grid.y, set[n].sz_grid.z );

    set[n].t_grid.x = set[n].iblk;
    set[n].t_grid.y = set[n].kblk;
    set[n].t_grid.z = 1;

    printf("Device %i: Allocating %i MB for grid ...", set[n].id, set[n].memsize/1048576);
    CudaSafeCall( cudaMalloc( (void**)&set[n].rings, set[n].memsize ) );
    CudaSafeCall( cudaMallocHost( (void**)&set[n].h_rings, set[n].memsize ) );
    printf(" done\n");

    #if ndim==3
    printf("Device %i: Allocating %i MB for boundaries ...", set[n].id, (2*n_pad*imax*sizeof(hydr_ring)+4*kmax*sizeof(hydr_ring))/1048576);
    CudaSafeCall( cudaMalloc( (void**)&set[n].udr, n_pad*imax*sizeof(hydr_ring) ) );
    CudaSafeCall( cudaMalloc( (void**)&set[n].top, n_pad*imax*sizeof(hydr_ring) ) );
    #else
    printf("Device %i: Allocating %i MB for boundaries ...", set[n].id, (4*kmax*sizeof(hydr_ring))/1048576);
    #endif
    CudaSafeCall( cudaMalloc( (void**)&set[n].lft, n_pad*kmax*sizeof(hydr_ring) ) );
    CudaSafeCall( cudaMalloc( (void**)&set[n].rgh, n_pad*kmax*sizeof(hydr_ring) ) );
    CudaSafeCall( cudaMalloc( (void**)&set[n].cp_lft, n_pad*kmax*sizeof(hydr_ring) ) );
    CudaSafeCall( cudaMalloc( (void**)&set[n].cp_rgh, n_pad*kmax*sizeof(hydr_ring) ) );
    printf(" done\n");

    #if file_flag==1
    CudaSafeCall( cudaMalloc( (void**)&set[n].d_output1, sizeof(sdp)*jmax*set[n].iblk*set[n].kblk ) );
    CudaSafeCall( cudaMallocHost( (void**)&set[n].h_output1, sizeof(sdp)*jmax*set[n].iblk*set[n].kblk ) );

    CudaSafeCall( cudaMalloc( (void**)&set[n].d_output2, sizeof(sdp)*jmax*set[n].iblk*set[n].kblk ) );
    CudaSafeCall( cudaMallocHost( (void**)&set[n].h_output2, sizeof(sdp)*jmax*set[n].iblk*set[n].kblk ) );

    CudaSafeCall( cudaMalloc( (void**)&set[n].d_output3, sizeof(sdp)*jmax*set[n].iblk*set[n].kblk ) );
    CudaSafeCall( cudaMallocHost( (void**)&set[n].h_output3, sizeof(sdp)*jmax*set[n].iblk*set[n].kblk ) );

    CudaSafeCall( cudaMalloc( (void**)&set[n].d_output4, sizeof(sdp)*jmax*set[n].iblk*set[n].kblk ) );
    CudaSafeCall( cudaMallocHost( (void**)&set[n].h_output4, sizeof(sdp)*jmax*set[n].iblk*set[n].kblk ) );

    CudaSafeCall( cudaMalloc( (void**)&set[n].d_output5, sizeof(sdp)*jmax*set[n].iblk*set[n].kblk ) );
    CudaSafeCall( cudaMallocHost( (void**)&set[n].h_output5, sizeof(sdp)*jmax*set[n].iblk*set[n].kblk ) );

    cout << "  Allocated " << 5*sizeof(sdp)*imax*jmax*kmax/1024/1024/nDev << "MB for output" << endl;
    #endif

    CudaSafeCall( cudaMalloc( (void**)&set[n].val, imax*kmax*sizeof(SymDisk) ) );

    CudaSafeCall( cudaMalloc( (void**)&set[n].dt, sizeof(sdp) ) );
    CudaSafeCall( cudaMallocHost( (void**)&set[n].h_dt, sizeof(sdp) ) );
    CudaSafeCall( cudaMallocHost( (void**)&set[n].dt_1D, sizeof(sdp)*set[n].iblk ) );
    CudaSafeCall( cudaMallocHost( (void**)&set[n].dt_2D, sizeof(sdp)*set[n].iblk*set[n].iblk ) );


    CudaSafeCall( cudaMalloc( (void**)&set[n].d_output, sizeof(sdp) ) );
    CudaSafeCall( cudaMallocHost( (void**)&set[n].h_output, sizeof(sdp) ) );
    cout << endl;
  }

  hydr_ring *lft = new hydr_ring[n_pad*kmax];
  hydr_ring *rgh = new hydr_ring[n_pad*kmax];
  hydr_ring *udr = new hydr_ring[n_pad*imax];
  hydr_ring *top = new hydr_ring[n_pad*imax];
  SymDisk *val = new SymDisk[imax*kmax];

  zxa=new sdp[imax];
  zxc=new sdp[imax];
  zdx=new sdp[imax];

  zya=new sdp[jmax];
  zyc=new sdp[jmax];
  zdy=new sdp[jmax];

  zza=new sdp[kmax];
  zzc=new sdp[kmax];
  zdz=new sdp[kmax];

  syncdevices(set);

//-----------------------------------------------------------------------------------------
// Generate Grid

  int npic = startat;

  if (restart)
  {
    ifstream sfile;
    open_binary_file(sfile, bfname+frame_num(npic));
    if(!sfile)
    {
      cout << endl << " Looking for " << bfname+frame_num(npic) << endl;
      cout << endl << " Restart data does not exist." << endl;
      return;
    }
    else
    {
      cout << endl << " Restarting from t=" << tmovie*npic << endl;
    }
    copy_bfile(sfile, set);
    copy_grid(set);
  }
  else 
  {
    init_grid();
  }

//-----------------------------------------------------------------------------------------
// Initializing
  sdp dt = endtime;

  sdp simtime = tmovie*(double)startat;
  sdp timep   = 0.0;
  sdp timem   = 0.0;
  int ncycle  = 0;
  int ncycp   = 0;

  init_den(val);
  cout << " density is set" << endl;

  init_speed(val);
  cout << " speed is set" << endl;

  if (!restart) init_cells(set, val);
  init_bound(lft, rgh, udr, top);
  cout << " boundary is set" << endl;

  //add_atmosphere(set);

  for (int n=0; n<nDev; n++)
  {
    CudaSafeCall( cudaMemcpyAsync( set[n].rings, set[n].h_rings, set[n].memsize, cudaMemcpyHostToDevice ) );

    CudaSafeCall( cudaMemcpyAsync( set[n].lft, lft, n_pad*kmax*sizeof(hydr_ring), cudaMemcpyHostToDevice ) );
    CudaSafeCall( cudaMemcpyAsync( set[n].rgh, rgh, n_pad*kmax*sizeof(hydr_ring), cudaMemcpyHostToDevice ) );

    #if ndim == 3
    CudaSafeCall( cudaMemcpyAsync( set[n].udr, udr, n_pad*imax*sizeof(hydr_ring), cudaMemcpyHostToDevice ) );
    CudaSafeCall( cudaMemcpyAsync( set[n].top, top, n_pad*imax*sizeof(hydr_ring), cudaMemcpyHostToDevice ) );
    #endif

    CudaSafeCall( cudaMemcpyAsync( set[n].val, val, imax*kmax*sizeof(SymDisk), cudaMemcpyHostToDevice ) );
  }
  syncdevices(set);

//-----------------------------------------------------------------------------------------
  #if dump_flag==1
  if (!restart)
  {
    open_output_file(pfile, pfname);
    write_para_file(pfile, mainp, label);
    pfile.close();

    open_binary_file(bfile, bfname+frame_num(npic));
    save_cells(bfile, set);
    bfile.close();
  }
  open_output_file(ifile, ifname);
  write_initial_file(ifile, zxa, zdx);
  ifile.close();

  if (!restart) open_output_file(tfile, tfname);
  else        append_output_file(tfile, tfname);
  #endif

  #if file_flag==1
  open_binary_file(result_file, result_fname);
  for (int n=0; n<nDev; n++)
  {
    CudaSafeCall( cudaSetDevice(set[n].id) );
    clear_output<<< kmax , imax, 0 , set[n].stream >>>(set[n].d_output1, set[n].d_output2, set[n].d_output3, set[n].d_output4, set[n].d_output5);
  }

  result *a = new result;
  (*a).q_ratio = M_p;
  (*a).aspect_ratio = sc_h;
  (*a).alpha = ss_alpha;
  #endif

  body planet;
  planet.m = M_p;
  planet.x = R_p;
  planet.y = pi;
  planet.vx = 0.0;
  planet.vy = pow(planet.x,-1.5)-FrRot;

/*
  //for (int i=0; i<n_pad; i++)   printf("%f : %.12f \n", lft[i+n_pad*0].xc, lft[i+n_pad*0].r[20] );
  //for (int i=0; i<imax; i++)   printf("%f : %.12f \n", set[0].h_rings[i+imax*0].xc, set[0].h_rings[i+imax*0].r[20] );
  //for (int i=0; i<n_pad; i++)   printf("%f : %.12f \n", rgh[i+n_pad*0].xc, rgh[i+n_pad*0].r[20] );
  //wait_f_r();
  
  sdp tmp;
  if (ndim == 3) tmp = (ymax-ymin) * (cos(zmin)-cos(zmax)) * third*(pow(xmax,3)-pow(xmin,3));
  else           tmp = (ymax-ymin) * 0.5*(pow(xmax,2)-pow(xmin,2));
  printf("True grid volume: %.12f \n", tmp );

  tmp=0.0;
  for (int i=0; i<imax; i++)
  {
    for (int k=0; k<kmax; k++)
    {
      for (int j=0; j<jmax; j++)
      {
        if (ndim == 3) tmp += (cos(zza[k])-cos(zza[k]+zdz[k])) * zdx[i]*(zxa[i]*(zxa[i] + zdx[i]) + zdx[i]*zdx[i]*third) * zdy[j];
        else           tmp += set[0].h_rings[i].xvol * set[0].h_rings[i].yvol[j];//zdx[i]*(zxa[i] + zdx[i]*0.5) * zdy[j];
      }
    }
  }
  printf("CPU volume sum: %.12f \n", tmp );

  tmp = GPU_output_reduction(set, planet, 0.0);
  printf("GPU volume sum: %.12f \n", tmp );


  dt = courant * (twopi/(sdp)jmax) / (sc_h/zxc[0] + pow(zxc[0],-1.5) - pow(0.75,-1.5));
  printf("Analytic dt with/o FARGO: %.12f \n", dt );

  dt=1.0;
  for (int i=0; i<imax; i++)
  {
    for (int k=0; k<kmax; k++)
    {
      for (int j=0; j<jmax; j++)
      {
        if (ndim == 3) dt = min(dt, courant*zxc[i]*sin(zzc[k])*zdy[j]/(sc_h) );
        else           dt = min(dt, courant*zxc[i]*zdy[j]/(sc_h) );
        dt = min(dt, courant*zdx[i]/(sc_h + val[i+imax*k].u) );
        if (ndim == 3) dt = min(dt, courant*zxc[i]*zdz[k]/(sc_h + val[i+imax*k].w) );
      }
    }
  }
  printf("CPU dt with FARGO: %.12f \n", dt );
*/
//============================================================================================
//                         MAIN COMPUTATIONAL LOOP

  clock_t begin, elapse;
  double speed;
  bool make_pict = false;
  bool prin_info = false;
  bool cont_simu = true;
  sdp old_dt;
  begin = clock();

  while (cont_simu)
  {
    dt = get_dt(dt, set);
    Index_Shift(set, dt);
    planet.m = set_M_p(simtime);
   
    if ( ncycle == ncycend )
    {
      prin_info = true;
      cont_simu = false;
    }
    if ( simtime + dt >= endtime )  // set dt to land on endtime
    {
      cout << "cutting to the end..." << " " << ncycle << " " << ncycend << endl;
      old_dt = dt;
      dt = endtime - simtime;
      prin_info = true;
      make_pict = true;
      cont_simu = false;
    }
    if ( timem + dt >= tmovie ) // set dt to land on tmovie
    {
      old_dt = dt;
      dt = tmovie - timem;
      make_pict = true;
    }

    #if ndim == 2
    bundle_sweep2D(set, dt, planet);
    #elif ndim == 3
    bundle_sweep3D(set, dt, planet);
    #endif
    planet.y += dt*(1.0-FrRot);
    if (planet.y<0.0)    planet.y += twopi;
    if (planet.y>=twopi) planet.y -= twopi;

    #if file_flag==1
    for (int n=0; n<nDev; n++) 
    {
      CudaSafeCall( cudaSetDevice(set[n].id) );
      cal_output<<< set[n].s_grid , idim, 0 , set[n].stream >>>(set[n].cells, dt, set[n].d_output1, set[n].d_output2, set[n].d_output3, set[n].d_output4, set[n].d_output5);
    }
    #endif

    simtime += dt;
    timem   += dt;
    ncycle++;
    ncycp++;

    if ( ncycp == nprin )
    {
      ncycp = 0;
      prin_info = true;
    }

    if (make_pict)
    {
      #if dump_flag == 1
      for (int n=0; n<nDev; n++)
      {
        CudaSafeCall( cudaMemcpy( set[n].h_rings, set[n].rings, set[n].memsize, cudaMemcpyDeviceToHost ) );
      }
      npic++;
      open_binary_file(bfile, bfname+frame_num(npic));
      save_cells(bfile, set);
      bfile.close();
      cout << " check point " << npic << " saved at " << bfname+frame_num(npic) << endl;
      #endif

      timem = 0.0;
      make_pict = false;
      dt = old_dt;
    }

    if (prin_info)
    {
      #if dump_flag==1
      tfile << simtime << " " << GPU_output_reduction(set, planet, 0.0)*sc_h*sc_h/M_p/M_p << endl;
      #endif

      elapse = clock()-begin;
      speed = (double)elapse/(double)CLOCKS_PER_SEC/(double)ncycle;
      cout << endl;
      cout << "# of steps = " << ncycle << " ( t/T = "<< simtime/endtime << ", dt = " << dt << " )" << endl;
      cout << "Average Speed is " << speed << " seconds per time step." << endl;
      cout << "Estimated time for completion: " << (speed*(double)ncycle)/60.0 << " of "
           << ((endtime-tmovie*(double)startat)*speed*(double)ncycle/(simtime-tmovie*(double)startat))/60.0 << " minutes." << endl;
      cout << endl;
      prin_info = false;
    }
  }
/*
  for (int n=0; n<nDev; n++)
  {
    CudaSafeCall( cudaMemcpy( set[n].h_rings, set[n].rings, set[n].memsize, cudaMemcpyDeviceToHost ) );
  }

  for (int i=0; i<n_pad; i++)  printf("%f : %.12e \n", lft[i].xc, lft[i].r[20]*cpow(lft[i].xc,p_alpha)-1.0 );
  for (int i=0; i<imax; i++)   printf("%f : %.12e \n", set[0].h_rings[i].xc, set[0].h_rings[i].r[20]*cpow(set[0].h_rings[i].xc,p_alpha)-1.0 );
  for (int i=0; i<n_pad; i++)  printf("%f : %.12e \n", rgh[i].xc, rgh[i].r[20]*cpow(rgh[i].xc,p_alpha)-1.0 );
  wait_f_r();
*/
  //P2P_all_disable(set);

  #if file_flag == 1
  for (int n=0; n<nDev; n++)
  {
    CudaSafeCall( cudaSetDevice(set[n].id) );
    CudaSafeCall( cudaMemcpy( set[n].h_output1, set[n].d_output1, sizeof(sdp)*imax*jmax*kmax/nDev, cudaMemcpyDeviceToHost ) );
    CudaSafeCall( cudaMemcpy( set[n].h_output2, set[n].d_output2, sizeof(sdp)*imax*jmax*kmax/nDev, cudaMemcpyDeviceToHost ) );
    CudaSafeCall( cudaMemcpy( set[n].h_output3, set[n].d_output3, sizeof(sdp)*imax*jmax*kmax/nDev, cudaMemcpyDeviceToHost ) );
    CudaSafeCall( cudaMemcpy( set[n].h_output4, set[n].d_output4, sizeof(sdp)*imax*jmax*kmax/nDev, cudaMemcpyDeviceToHost ) );
    CudaSafeCall( cudaMemcpy( set[n].h_output5, set[n].d_output5, sizeof(sdp)*imax*jmax*kmax/nDev, cudaMemcpyDeviceToHost ) );
  }

  int n;
  int ind;
  (*a).start_t = tmovie*(double)startat;
  (*a).end_t = simtime;

  for (int i=0; i<imax; i++)
  {
    (*a).x[i]  = zxc[i];
    (*a).dx[i] = zdx[i];
  }

  for (int j=0; j<jmax; j++)
  {
    (*a).y[j]  = zyc[j];
    (*a).dy[j] = zdy[j];
  }

  for (int k=0; k<kmax; k++)
  {
    (*a).z[k]  = zzc[k];
    (*a).dz[k] = zdz[k];
  }

  for (int i=0; i<imax; i++)
    for (int j=0; j<jmax; j++)
    {
      n = j*nDev/jmax;
      for (int k=0; k<kmax; k++)
      {
        ind = i+imax*((j-n*jmax/nDev)+(jmax/nDev)*k);
        (*a).r[i][j][k] = set[n].h_output1[ind]/(simtime - tmovie*(double)startat);
        (*a).p[i][j][k] = set[n].h_output2[ind]/set[n].h_output1[ind];
        (*a).u[i][j][k] = set[n].h_output3[ind]/set[n].h_output1[ind];
        (*a).v[i][j][k] = set[n].h_output4[ind]/set[n].h_output1[ind];
        (*a).w[i][j][k] = set[n].h_output5[ind]/set[n].h_output1[ind];
      }
    }
  result_file.write((char*)a, sizeof(result));
  result_file.close();
  #endif
 
  for (int n=0; n<nDev; n++)
  {
    CudaSafeCall( cudaSetDevice(set[n].id) );
    CudaSafeCall( cudaFree( set[n].rings ) );
    CudaSafeCall( cudaFreeHost( set[n].h_rings ) );

    CudaSafeCall( cudaFree( set[n].lft ) );
    CudaSafeCall( cudaFree( set[n].rgh ) );
    CudaSafeCall( cudaFree( set[n].cp_lft ) );
    CudaSafeCall( cudaFree( set[n].cp_rgh ) );
    #if ndim==3
    CudaSafeCall( cudaFree( set[n].udr ) );
    CudaSafeCall( cudaFree( set[n].top ) );
    #endif

    #if file_flag==1
    CudaSafeCall( cudaFree( set[n].d_output1 ) );
    CudaSafeCall( cudaFreeHost( set[n].h_output1 ) );

    CudaSafeCall( cudaFree( set[n].d_output2 ) );
    CudaSafeCall( cudaFreeHost( set[n].h_output2 ) );

    CudaSafeCall( cudaFree( set[n].d_output3 ) );
    CudaSafeCall( cudaFreeHost( set[n].h_output3 ) );

    CudaSafeCall( cudaFree( set[n].d_output4 ) );
    CudaSafeCall( cudaFreeHost( set[n].h_output4 ) );

    CudaSafeCall( cudaFree( set[n].d_output5 ) );
    CudaSafeCall( cudaFreeHost( set[n].h_output5 ) );
    #endif

    CudaSafeCall( cudaFree( set[n].val ) );

    CudaSafeCall( cudaFree( set[n].dt ) );
    CudaSafeCall( cudaFreeHost( set[n].h_dt ) );

    CudaSafeCall( cudaFree( set[n].d_output ) );
    CudaSafeCall( cudaFreeHost( set[n].h_output ) );
    CudaSafeCall( cudaDeviceReset() );
  }
  delete[] lft,rgh,udr,top,val;
  #if file_flag == 1
  delete[] a;
  #endif
//                           END OF MAIN LOOP
//============================================================================================
  return;
}

int main(int narg, char *args[])
{
  //signal(SIGINT,int_handler);
  double ttau = 1.0;
  sdp B0 = 0.2;

  #if opac_flag == 0
  B0 = 0.0;
  #endif

  bool restart = false;
  int morph = 0;
  string morph_path;
  
  if      (narg==1)
  {
    restart = false;
    kernel(restart, 0, 0, morph_path, B0, ttau);
  }
  else if (narg==2)
  {
    if (string(args[1])=="init")
    {
      show_init(B0, ttau);
    }
    else
    {
      restart = true;
      kernel(restart, atof(args[1]), 0, morph_path, B0, ttau);
    }
  }
  else if (narg==3) 
  {
    restart = false; 
    morph = atof(args[2]); 
    morph_path = string(args[1]);
    kernel(restart, 0, morph, morph_path, B0, ttau);
  }     

  return 0;
}
