/*
 * Placeholder OpenCL kernel
 */

__kernel void find_max(__global unsigned int *data, const unsigned int length)
{
  unsigned int pos = 0;
  unsigned int val;
  unsigned int id = get_global_id(0);
  unsigned int size = get_global_size(0);
  pos = length/size*id;

  //printf("size = %d, length = %d, pos = %d\n", size, length, pos);
  val = data[pos];
  for(pos; pos < length/size * (id + 1); pos++){
  if(data[pos] > val){
      val = data[pos];
    }
  }
  //printf("thread id = %d, val = %d\n", id, val);
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  data[id] = val;
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  val = data[0];
  if(id==0){
      for(int i = 0; i < size; i++){
        if(data[i] > val){
          val = data[i];
        }
      }
    data[0]=val;
  }
}
