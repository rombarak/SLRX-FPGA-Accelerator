#ifndef _POOL_H_
#define _POOL_H_

#include <k5_libs.h>
#include <slr_lib.h>
#include "slrx.h" 

//------------------------------------------------------------------------------------------------------------

void pool_xlr_setup(uint8_t* pool_arr_out, // Pool output feature-map
                    uint8_t* pool_arr_in,  // Pool Input Image
                    int      arr_in_dim);  // Pool Input array dimensions                    


//------------------------------------------------------------------------------------------------------------


void pool_max_2x2(uint8_t* pool_arr_out, // Pool-Max output feature-map
                  uint8_t* pool_arr_in,  // Pool-Max Input Image  
                  int     arr_in_dim);   // Pool-Max Input dimensions 

#endif