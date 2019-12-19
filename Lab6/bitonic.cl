/*
 * Placeholder OpenCL kernel
 */

__kernel void bitonic(__global unsigned int *data, const unsigned int length, const int k, const int j)
{
const int i = get_global_id(0);
int ixj=i^j; // Calculate indexing!
if ((ixj)>i)
  {
    if ((i&k)==0 && data[i]>data[ixj]){
      int tmp = data[i];
      data[i] = data[ixj];
      data[ixj] = tmp;
    }
    if ((i&k)!=0 && data[i]<data[ixj]){
      int tmp = data[i];
      data[i] = data[ixj];
      data[ixj] = tmp;
    }
  }
}
