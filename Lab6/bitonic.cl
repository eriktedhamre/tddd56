/*
 * Placeholder OpenCL kernel
 */

__kernel void bitonic(__global unsigned int *data, const unsigned int length)
{
  unsigned int pos = 0;
  unsigned int val;
  unsigned int id = get_global_id(0);
  unsigned int size = get_global_size(0);
  unsigned int local_size = get_local_size(0);
  unsigned int local_id = get_local_id(0);
  unsigned int group_id = get_group_id(0);

  unsigned int i,j,k;

  for (k=2;k<=length;k=2*k) // Outer loop, double size for each step
  {
    for (j=k>>1;j>0;j=j>>1) // Inner loop, half size for each step
    {
      //for (i=0;i<length;i++) // Loop over data
      int i = id;
      int ixj=i^j; // Calculate indexing!
      if ((ixj)>i)
      {
        //printf("id=%d, j=%d, k=%d, i=%d, ixj=%d, i&k=%d, data[i]=%d, data[ixj]=%d\n", id, j, k, i, ixj, i&k, data[i], data[ixj]);
        if ((i&k)==0 && data[i]>data[ixj])
        {
          int tmp = data[i];
          data[i] = data[ixj];
          data[ixj] = tmp;
        }
        if ((i&k)!=0 && data[i]<data[ixj])
        {
          int tmp = data[i];
          data[i] = data[ixj];
          data[ixj] = tmp;
        }
        //printf("Second id=%d, j=%d, k=%d, i=%d, ixj=%d, i&k=%d, data[i]=%d, data[ixj]=%d\n", id, j, k, i, ixj, i&k, data[i], data[ixj]);
      }

      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      i = length-id-1;
      ixj=i^j; // Calculate indexing!
      if ((ixj)>i)
      {
        //printf("Third id =%d, j=%d, k=%d, i=%d, ixj=%d, i&k=%d, data[i]=%d, data[ixj]=%d\n", id, j, k, i, ixj, i&k, data[i], data[ixj]);
        if ((i&k)==0 && data[i]>data[ixj])
        {
          int tmp = data[i];
          data[i] = data[ixj];
          data[ixj] = tmp;
        }
        if ((i&k)!=0 && data[i]<data[ixj])
        {
          int tmp = data[i];
          data[i] = data[ixj];
          data[ixj] = tmp;
        }
        //printf("Fourth id =%d, j=%d, k=%d, i=%d, ixj=%d, i&k=%d, data[i]=%d, data[ixj]=%d\n", id, j, k, i, ixj, i&k, data[i], data[ixj]);
      }
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
  }
}
