#include <k5_libs.h>
#include <slr_lib.h>
#include "slrx.h" 

//------------------------------------------------------------------------------------------------------------

void pool_xlr_setup(uint8_t* pool_arr_out,  // Pool output feature-map
                    uint8_t* pool_arr_in,   // Pool Input Image
                    int      arr_in_dim) {  // Pool Input array dimensions                    

    #ifdef HLCM    
    printf("HLCM does not support HW acceleration, quitting\n\n");
    bm_quit_app();
    #else

    HOST_REG(ARR_IN_ADDR_RI)  = (unsigned int)pool_arr_in;
    HOST_REG(ARR_OUT_ADDR_RI) = (unsigned int)pool_arr_out;
    HOST_REG(ARR_IN_DIM_RI)   = arr_in_dim;
                              
    HOST_REG(XLR_START_RI) = POOL_SETUP ; // POOL_SETUP is defined at included ../../../hw/top/slrx_enums.svh
    
    while (!HOST_REG(XLR_DONE_RI)) {
       //printf("Pool setup Polling ...\n"); // comment for quite execution
    }
    
    #endif 
}

//------------------------------------------------------------------------------------------------------------

void pool_window_xlr(int out_row_idx) { // output array row index

    #ifdef HLCM    
    printf("HLCM does not support HW acceleration, quitting\n\n");
    bm_quit_app();
    #else

    HOST_REG(OUT_ROW_IDX_RI) = out_row_idx; 
    HOST_REG(XLR_START_RI)  = POOL_CALC ; // POOL_CALC is defined at included ../../../hw/top/slrx_enums.svh
  
    while (!HOST_REG(XLR_DONE_RI)) {
       //printf("Pool window Polling ...\n"); // comment for quite execution
    }
    
    #endif 
}

//------------------------------------------------------------------------------------------------------------

void pool_window_nox(uint8_t* pool_arr_out,     // Pool-Max output feature-map
                     uint8_t* pool_arr_in,      // Pool-Max Input Image  
                     int      out_row_idx,      // Pool Output row index
                     int      pool_out_dim) {   // Pool-Max Output dimensions 

    int in_row0_idx = out_row_idx * 2;
    int in_row1_idx = in_row0_idx + 1;
    int pool_in_dim = pool_out_dim * 2 ;
    
    for (int out_col_idx = 0; out_col_idx < pool_out_dim; out_col_idx++) {

        int in_col0_idx = out_col_idx * 2;
        int in_col1_idx = in_col0_idx + 1;

        uint8_t val0 = pool_arr_in[(in_row0_idx * pool_in_dim) +in_col0_idx] ;
        uint8_t val1 = pool_arr_in[(in_row0_idx * pool_in_dim) +in_col1_idx] ;
        uint8_t val2 = pool_arr_in[(in_row1_idx * pool_in_dim) +in_col0_idx] ;
        uint8_t val3 = pool_arr_in[(in_row1_idx * pool_in_dim) +in_col1_idx] ;

        uint8_t max01 = val0 > val1 ? val0 : val1 ;
        uint8_t max23 = val2 > val3 ? val2 : val3 ;  
         
        uint8_t max_pool = max01 > max23 ? max01 : max23 ;

        ((volatile uint8_t*)pool_arr_out)[(out_row_idx * pool_out_dim) + out_col_idx] = max_pool;
    }
}

//------------------------------------------------------------------------------------------------------------

void pool_max_2x2(uint8_t* pool_arr_out,         // Pool-Max output feature-map
                  uint8_t* pool_arr_in,          // Pool-Max Input Image  
                  int     arr_in_dim) {    // Pool-Max Input dimensions 

    int pool_out_dim = arr_in_dim/2;

      #ifdef POOL_XON
      pool_xlr_setup(pool_arr_out, pool_arr_in, arr_in_dim);              
      #endif

    for (int out_row_idx = 0; out_row_idx < pool_out_dim; out_row_idx++) {
      #ifdef POOL_XON
      pool_window_xlr(out_row_idx);
      #else
      pool_window_nox(pool_arr_out,  pool_arr_in, out_row_idx, pool_out_dim);
      #endif  
    }      
}

