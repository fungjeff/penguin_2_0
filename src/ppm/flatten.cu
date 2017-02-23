//===============================================================================
//use the numbers below for more aggresive flattening
#define flat_omega1 0.75    // 0.5    //0.75
#define flat_omega2 5.0     // 10.0   //5.0
#define flat_epsilon 0.33   // 1.0    //0.33

__device__ void flatten(int &n, int nmax, sdp *p, sdp *u, sdp *steep, sdp &flat)
{
  sdp dp1, dp2, ddp;

  bool shock = false;
  steep[n] = 0.0;
  if (n>=2 && n<nmax-2)
  {
    dp1 = p[n+1] - p[n-1];
    dp2 = p[n+2] - p[n-2];
    if(cabs(dp2) < smallp) dp2 = smallp;

    ddp = cabs(dp1)/cmin(p[n+1],p[n-1]);
 
    if(ddp>flat_epsilon) shock = true;

    if(u[n-1] < u[n+1]) shock = false;

    if(shock)
    {
      ddp = ( dp1 / dp2 - flat_omega1 ) * flat_omega2;
      steep[n] = cmax( 0.0, ddp );
    }
  }
  __syncthreads();

  if (n>=2 && n<nmax-2)
  {
    dp1  = max3( steep[n-1], steep[n], steep[n+1] );
    flat = cmin( 0.5, dp1 );
  }
  __syncthreads();
  return;
}
