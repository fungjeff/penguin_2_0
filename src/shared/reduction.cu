//=========================================================================

__device__ int two_round(int x)
{
  int y = 1;
  while (y<x)
    y *= 2;
  return y;
}

__device__ int log2(int x)
{
  int y = 0;
  while (x>1)
  {
    x /= 2;
    y++;
  }
  return y;
}

__device__ void shift_round_reduc_max(int nmin, int len, sdp *list)
{

  int halfPoint, thread2;
  int len2 = two_round(len); // Total number of threads, rounded up to the next power of two
  int n = threadIdx.x;

  while(len2 > 1)
  {
    halfPoint = len2/2;
    // only the first half of the threads will be active.
 
    if (n < halfPoint)
    {
      thread2 = n + halfPoint;
 
      // Skipping the fictious threads blockDim.x ... blockDim_2-1
      if (thread2 < len)
      {
        if (list[n+nmin]<list[thread2+nmin]) list[n+nmin] = list[thread2+nmin];
      }
    }
    __syncthreads();
 
    // Reducing the binary tree size by two:
    len2 = halfPoint;
  }
  return;
}

__device__ void round_reduc_max(int len, sdp *list)
{
  int halfPoint, thread2;
  int len2 = two_round(len);
  int n = threadIdx.x + blockDim.x*threadIdx.y;

  while(len2 > 1)
  {
    halfPoint = len2/2;
 
    if (n < halfPoint)
    {
      thread2 = n + halfPoint;
      if (thread2 < len)
      {
        if (list[n]<list[thread2]) list[n] = list[thread2];
      }
    }
    __syncthreads();
 
    len2 = halfPoint;
  }
  return;
}

__device__ void round_reduc_sum(int len, sdp *list)
{
  int halfPoint, thread2;
  int len2 = two_round(len);
  int n = threadIdx.x + blockDim.x*threadIdx.y;

  while(len2 > 1)
  {
    halfPoint = len2/2;
 
    if (n < halfPoint)
    {
      thread2 = n + halfPoint;
      if (thread2 < len)
      {
        list[n] += list[thread2];
      }
    }
    __syncthreads();
 
    len2 = halfPoint;
  }
  return;
}

__device__ void bin_reduc_max(int len, sdp *list)
{
  int halfPoint, thread2;
  int n = threadIdx.x + blockDim.x*threadIdx.y;

  while(len > 1)
  {
    halfPoint = len/2;
 
    if (n < halfPoint)
    {
      thread2 = n + halfPoint;
      if (list[n]<list[thread2]) list[n] = list[thread2];
    }
    __syncthreads();
 
    len = halfPoint;
  }
  return;
}

__device__ void bin_reduc_sum(int len, sdp *list)
{
  int halfPoint, thread2;
  int n = threadIdx.x + blockDim.x*threadIdx.y;

  while(len > 1)
  {
    halfPoint = len/2;
 
    if (n < halfPoint)
    {
      thread2 = n + halfPoint;
      list[n] += list[thread2];
    }
    __syncthreads();
 
    len = halfPoint;
  }
  return;
}
