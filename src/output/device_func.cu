//#########################################################################################

__global__ void clear_output(sdp *out1, sdp *out2, sdp *out3, sdp *out4, sdp *out5)
{
  int i = blockIdx.x;
  int k = blockIdx.y;
  int j = threadIdx.x + blockDim.x * blockIdx.z;
  int ind;

  if (j<jmax)
  {
    ind = i + gridDim.x * (j + (jmax * k));
    out1[ind] = 0.0;
    out2[ind] = 0.0;
    out3[ind] = 0.0;
    out4[ind] = 0.0;
    out5[ind] = 0.0;
  }

  return;
}

__global__ void cal_output(hydr_ring *rings, sdp weight, sdp *out1, sdp *out2, sdp *out3, sdp *out4, sdp *out5)
{
  int i = blockIdx.x;
  int k = blockIdx.y;
  int j = threadIdx.x + blockDim.x * blockIdx.z;
  int ind;
  int ik = i + gridDim.x * k;
  sdp dt_rho;

  if (j<jmax)
  {
    ind = i + gridDim.x * (j + (jmax * k));
    dt_rho = weight*rings[ik].r[j];

    out1[ind] += dt_rho;
    out2[ind] += dt_rho*rings[ik].p[j];
    out3[ind] += dt_rho*rings[ik].u[j];
    out4[ind] += dt_rho*rings[ik].v[j];
    out5[ind] += dt_rho*rings[ik].w[j];
  }

  return;
}

//#########################################################################################

__global__ void output_lv1(hydr_ring *rings, sdp *dt_2D, body planet, sdp exclude, int mode)
{
  int i = blockIdx.x;
  int k = blockIdx.y;
  int j;
  int ik = i + gridDim.x * k;
  int tt = threadIdx.x;

  __shared__ sdp ridt[1024];
  sdp tmp, rad, z, phi, r_hill;

  int tmax = blockDim.x;
  int loop = jmax/tmax;
  if (jmax%tmax) loop++;
  sdp val = 0.0;

  for (int l=0; l<loop; l++)
  {
    j = tt + tmax * l;

    if (j<jmax)
    {
      rad = rings[ik].xc;
      phi = rings[ik].yc[j];
    
      #if plnt_flag == 1
      r_hill = cmax(cpow(M_p/3.0,1.0/3.0),sc_h);
      if (ndim == 2)
      {
        if ( (rad-1.0)*(rad-1.0) + (phi-pi)*(phi-pi) > exclude*exclude )
        {
          tmp  = -star_planet_grav_azi_cyl(rad, phi, 0.0, planet);
          tmp *= rad * rings[ik].r[j] * rings[ik].xvol * rings[ik].yvol[j] * rings[ik].zvol;
          ridt[tt] = tmp;
        }
        else
        {
          ridt[tt] = 0.0;
        }
      }
      else if (ngeomz == 0)
      {
        z = rings[ik].zc;
        if ( (rad-1.0)*(rad-1.0) + (phi-pi)*(phi-pi) + z*z > exclude*exclude )
        {
          tmp  = -star_planet_grav_azi_cyl(rad, phi, z, planet);
          tmp *= rad * rings[ik].r[j] * rings[ik].xvol * rings[ik].yvol[j] * rings[ik].zvol;
          ridt[tt] = tmp;
        }
        else
        {
          ridt[tt] = 0.0;
        }
      }
      else if (ngeomz == 5)
      {
        z = rad*ccos(rings[ik].zc);
        rad = csqrt(rad*rad-z*z);
        if ( (rad-1.0)*(rad-1.0) + (phi-pi)*(phi-pi) + z*z > exclude*exclude )
        {
          tmp  = -star_planet_grav_azi_cyl(rad, phi, z, planet);
          tmp *= rad * rings[ik].r[j] * rings[ik].xvol * rings[ik].yvol[j] * rings[ik].zvol;
          ridt[tt] = tmp;
        }
        else
        {
          ridt[tt] = 0.0;
        }
      }
      #else
      tmp = rings[ik].r[j] * rings[ik].xvol * rings[ik].yvol[j] * rings[ik].zvol;
      ridt[tt] = tmp;
      #endif
    }
    else
    {
      ridt[tt] = 0.0;  
    }
    __syncthreads();

    bin_reduc_sum(1024, ridt);

    if (tt==0) val+=ridt[0];
    __syncthreads();
  }
  if (tt==0) dt_2D[ik]=val;

  return;
}


__global__ void output_lv2(sdp *dt_2D, sdp *dt_1D)
{
  int i = blockIdx.x;
  int k = threadIdx.x;
  int ik = i + gridDim.x * k;

  __shared__ sdp ridt[kmax];

  ridt[k] = dt_2D[ik];
  __syncthreads();

  round_reduc_sum(kmax, ridt);

  if (k==0) dt_1D[i]=ridt[0];
  
  return;
}

__global__ void output_lv3(sdp *dt_1D, sdp *output, int iblk)
{
  int i = threadIdx.x;

  __shared__ sdp ridt[1024];
  if (i<iblk)  ridt[i] = dt_1D[i];
  else         ridt[i] = 0.0;
  __syncthreads();

  bin_reduc_sum(1024, ridt);

  if (i==0) *output = ridt[0];

  return;
}

sdp GPU_output_reduction(GPU_plan *set, body planet, sdp exclude, int mode)
{
  exclude *= pow((M_p/3.0),1.0/3.0);
  //exclude *= M_p/sc_h/sc_h;
  sdp total = 0.0;

  for (int n=0; n<nDev; n++)
  {
    CudaSafeCall( cudaSetDevice(set[n].id) );

    output_lv1<<< set[n].t_grid , 1024 , 0 , set[n].stream >>>(set[n].rings, set[n].dt_2D, planet, exclude, mode);
    CudaCheckError();
    output_lv2<<< set[n].iblk , set[n].kblk , 0 , set[n].stream >>>(set[n].dt_2D, set[n].dt_1D);
    CudaCheckError();
    output_lv3<<< 1 , 1024 , 0 , set[n].stream >>>(set[n].dt_1D, set[n].d_output, set[n].iblk);
    CudaCheckError();

    CudaSafeCall( cudaMemcpyAsync( set[n].h_output, set[n].d_output, sizeof(sdp), cudaMemcpyDeviceToHost, set[n].stream) );
  }

  for (int n=0; n<nDev; n++) CudaSafeCall( cudaStreamSynchronize(set[n].stream) );

  for (int n=0; n<nDev; n++) total += *set[n].h_output;

  return total;
}
