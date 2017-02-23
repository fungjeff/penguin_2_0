//////////////////////////////////////////////////////////////////////
__device__ sdp min3(sdp A, sdp B, sdp C)
{
  if(A<B)
  {
    if(C<A) return C;
    else return A;
  }
  else
  {
    if(C<B) return C;
    else return B;
  }
}

__device__ sdp max3(sdp A, sdp B, sdp C)
{
  if(A>B)
  {
    if(C>A) return C;
    else return A;
  }
  else
  {
    if(C>B) return C;
    else return B;
  }
}
//////////////////////////////////////////////////////////////////////
__device__ sdp cu_cos(sdp x)
{
  sdp out;
  sdp tmp = 1.0;
  sdp f   = 1.0;
  sdp x2  = -x*x;

  if (x<1.0 && x>-1.0)
  {
    out = tmp;
    for (int i=0; i<5; i++)
    {
      tmp *= x2/f/(f+1.0);
      out += tmp;
      f   += 2.0;
    }
    return out;
  }
  else
    return cosf(x);
}

__device__ sdp cu_sin(sdp x)
{
  sdp out;
  sdp tmp = x;
  sdp f   = 2.0;
  sdp x2  = -x*x;
  
  if (x<1.0 && x>-1.0)
  {
    out = tmp;
    for (int i=0; i<5; i++)
    {
      tmp *= x2/f/(f+1.0);
      out += tmp;
      f   += 2.0;
    }
    return out;
  }
  else
    return sinf(x);
}
