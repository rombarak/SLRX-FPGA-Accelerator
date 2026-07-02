#include <k5_libs.h>
#include <slr_lib.h>
#include "slrx.h" 

//---------------------------------------------------------------------------------------------------------------------------------

// Compute a single 5x5 conv window and RETURN the activated result (relu_and_descale,
// uint8) instead of storing it. This is the single source of truth for the conv MAC:
// both conv_window_nox (separate path) and conv_pool_fused_nox (fused golden model) call
// it, so the two paths are bit-identical by construction.
uint8_t conv_window_val_nox(uint8_t* conv_arr_in,                         // Conv Input Image
                            int      arr_in_dim,                          // Conv Input array dimensions
                            int      out_row_idx,                         // output array row index
                            int      out_col_idx,                         // output array column index
                            int8_t*  kernel_w,                            // Conv kernel Weights, can be negative
                            int32_t  kernel_b) {                          // Conv kernel Bias, can be negative

    int32_t acc = kernel_b;

    for (int kernel_row_idx = 0; kernel_row_idx < CONV_KERNEL_DIM; kernel_row_idx++) {
        for (int kernel_col_idx = 0; kernel_col_idx < CONV_KERNEL_DIM; kernel_col_idx++) {

            int in_row_idx = out_row_idx + kernel_row_idx;
            int in_col_idx = out_col_idx + kernel_col_idx;

            int arr_in_idx = (in_row_idx * arr_in_dim) + in_col_idx ;

            uint8_t in_val = ((volatile uint8_t*)conv_arr_in)[arr_in_idx];
            int8_t weight  = ((volatile int8_t(*)[CONV_KERNEL_DIM])kernel_w)[kernel_row_idx][kernel_col_idx];

            acc += (int32_t)in_val * (int32_t)weight;
        }
    }
    // ReLU + descale (/256) + saturate to 255 -> uint8
    return relu_and_descale(acc);
}

//------------------------------------------------------------------------------------------------------------

void conv_window_nox(uint8_t* conv_arr_out,                               // Conv output feature-map
                     uint8_t* conv_arr_in,                                // Conv Input Image
                     int      arr_in_dim,                                 // Conv Input array dimensions
                     int      out_row_idx,                                // output array row index
                     int      out_col_idx,                                // output array column index
                     int8_t*  kernel_w, // kernel_w[CONV_KERNEL_DIM][CONV_KERNEL_DIM], // Conv kernel Weights, can be negative
                     int32_t  kernel_b) {                                 // Conv kernel Bias, can be negative

    //printf("%x,%x\n",out_row_idx,out_col_idx);// DBG

    int out_dim = arr_in_dim - CONV_KERNEL_DIM + 1;

    // store with saturation
    int arr_out_idx = (out_row_idx * out_dim) + out_col_idx;
    ((volatile uint8_t*)conv_arr_out)[arr_out_idx] =
        conv_window_val_nox(conv_arr_in, arr_in_dim, out_row_idx, out_col_idx, kernel_w, kernel_b);
}

//------------------------------------------------------------------------------------------------------------

void conv_xlr_setup(uint8_t* conv_arr_out,                               // Conv output feature-map
                    uint8_t* conv_arr_in,                                // Conv Input Image
                    int      arr_in_dim,                                 // Conv Input array dimensions                    
                    int8_t*  kernel_w, // int8_t   kernel_w[CONV_KERNEL_DIM][CONV_KERNEL_DIM], // Conv kernel Weights, can be negative
                    int32_t  kernel_b) {                                 // Conv kernel Bias, can be negative

    #ifdef HLCM    
    printf("HLCM does not support HW acceleration, quitting\n\n");
    bm_quit_app();
    #else
    HOST_REG(WGT_ADDR_RI) = (unsigned int)kernel_w;
    HOST_REG(CONV_BIAS_VAL_RI)        = kernel_b;
    HOST_REG(ARR_IN_ADDR_RI)          = (unsigned int)conv_arr_in;
    HOST_REG(ARR_OUT_ADDR_RI)         = (unsigned int)conv_arr_out;
    HOST_REG(ARR_IN_DIM_RI)           = arr_in_dim;

    HOST_REG(XLR_START_RI) = CONV_SETUP ; // CONV_SETUP is defined at included ../../../hw/top/slrx_enums.svh
    
    while (!HOST_REG(XLR_DONE_RI)) {
       //printf("Conv setup Polling ...\n"); // comment for quite execution
    }
    
    #endif 
}

//------------------------------------------------------------------------------------------------------------

void conv_window_xlr(int out_row_idx,   // output array row index
                     int out_col_idx){  // output array column index 

    #ifdef HLCM    
    printf("HLCM does not support HW acceleration, quitting\n\n");
    bm_quit_app();
    #else

    HOST_REG(OUT_ROW_IDX_RI)     = out_row_idx; 
    HOST_REG(OUT_COL_IDX_RI)     = out_col_idx; 
    HOST_REG(XLR_START_RI) = CONV_WINDOW ; // CONV_WINDOW is defined at included ../../../hw/top/slrx_enums.svh
  
    while (!HOST_REG(XLR_DONE_RI)) {
       //printf("Conv window Polling ...\n"); // comment for quite execution
    }
    
    #endif 
}

//------------------------------------------------------------------------------------------------------------


void conv(uint8_t* conv_arr_out,                               // Conv output feature-map
          uint8_t* conv_arr_in,                                // Conv Input Image  
          int      arr_in_dim,                                 // Conv Input dimensions  
          int8_t*  kernel_w, ///  int8_t   kernel_w[CONV_KERNEL_DIM][CONV_KERNEL_DIM], // Conv kernel Weights, can be negative
          int32_t  kernel_b) {                                 // Conv kernel Bias, can be negative

    int out_dim = arr_in_dim - CONV_KERNEL_DIM + 1;

    #ifdef CONV_XON
    conv_xlr_setup(conv_arr_out, conv_arr_in, arr_in_dim, kernel_w, kernel_b);       
    #endif
   
    for (int out_row_idx = 0; out_row_idx < out_dim; out_row_idx++){
      for (int out_col_idx = 0; out_col_idx < out_dim; out_col_idx++){

        #ifdef CONV_XON
        conv_window_xlr(out_row_idx, out_col_idx); // assume setup called once per execution
        #else
        conv_window_nox(conv_arr_out, conv_arr_in, arr_in_dim, out_row_idx, out_col_idx, kernel_w, kernel_b);
        #endif
      }
    }

}

//------------------------------------------------------------------------------------------------------------

// Fused Conv(5x5, no padding) + 2x2 MaxPool software golden model.
// Computes the conv feature-map on the fly and writes the POOLED output directly, with NO
// intermediate full conv feature-map stored in memory. Bit-identical to conv() followed by
// pool_max_2x2(): each pooled cell is the 2x2 max over the four conv windows that the
// separate pool would have read, and every window reuses conv_window_val_nox (same MAC +
// relu_and_descale), so the four maxed values are uint8 exactly as in the separate path.
void conv_pool_fused_nox(uint8_t* pool_arr_out,                           // pooled output feature-map (post_pX_fm)
                         uint8_t* conv_arr_in,                            // conv input image / feature-map
                         int      arr_in_dim,                             // conv input dimension
                         int8_t*  kernel_w,                               // 5x5 conv kernel weights, can be negative
                         int32_t  kernel_b) {                             // conv scalar bias, can be negative

    int conv_out_dim = arr_in_dim - CONV_KERNEL_DIM + 1; // conv feature-map dim (28 or 10)
    int pool_out_dim = conv_out_dim / 2;                 // pooled output dim   (14 or 5); truncating /2 matches pool

    for (int pool_row_idx = 0; pool_row_idx < pool_out_dim; pool_row_idx++) {
        for (int pool_col_idx = 0; pool_col_idx < pool_out_dim; pool_col_idx++) {

            // The 2x2 block of conv windows feeding this pooled cell
            int c_row0 = pool_row_idx * 2;   // top conv row
            int c_row1 = c_row0 + 1;         // bottom conv row
            int c_col0 = pool_col_idx * 2;   // left conv col
            int c_col1 = c_col0 + 1;         // right conv col

            uint8_t v00 = conv_window_val_nox(conv_arr_in, arr_in_dim, c_row0, c_col0, kernel_w, kernel_b);
            uint8_t v01 = conv_window_val_nox(conv_arr_in, arr_in_dim, c_row0, c_col1, kernel_w, kernel_b);
            uint8_t v10 = conv_window_val_nox(conv_arr_in, arr_in_dim, c_row1, c_col0, kernel_w, kernel_b);
            uint8_t v11 = conv_window_val_nox(conv_arr_in, arr_in_dim, c_row1, c_col1, kernel_w, kernel_b);

            // 2x2 max-pool over the four activated (uint8) window results
            uint8_t max01    = v00 > v01 ? v00 : v01;
            uint8_t max23    = v10 > v11 ? v10 : v11;
            uint8_t max_pool = max01 > max23 ? max01 : max23;

            ((volatile uint8_t*)pool_arr_out)[(pool_row_idx * pool_out_dim) + pool_col_idx] = max_pool;
        }
    }
}

//------------------------------------------------------------------------------------------------------------

// Fused Conv+MaxPool via the hardware accelerator. The accelerator is hardware-looped: a
// single CONV_WINDOW command runs the whole layer (it loops over all pooled positions
// internally, like the hardware-looped linear), so we issue once and poll once.
void conv_pool_fused_xlr(uint8_t* pool_arr_out,                           // pooled output feature-map
                         uint8_t* conv_arr_in,                            // conv input image / feature-map
                         int      arr_in_dim,                             // conv input dimension
                         int8_t*  kernel_w,                               // 5x5 conv kernel weights
                         int32_t  kernel_b) {                             // conv scalar bias

    #ifdef HLCM
    printf("HLCM does not support HW acceleration, quitting\n\n");
    bm_quit_app();
    #else
    HOST_REG(WGT_ADDR_RI)      = (unsigned int)kernel_w;
    HOST_REG(CONV_BIAS_VAL_RI) = kernel_b;
    HOST_REG(ARR_IN_ADDR_RI)   = (unsigned int)conv_arr_in;
    HOST_REG(ARR_OUT_ADDR_RI)  = (unsigned int)pool_arr_out;
    HOST_REG(ARR_IN_DIM_RI)    = arr_in_dim;

    HOST_REG(XLR_START_RI) = CONV_WINDOW ; // single fused full-layer command (defined in slrx_enums.svh)

    while (!HOST_REG(XLR_DONE_RI)) {
       //printf("Conv fused Polling ...\n"); // comment for quite execution
    }
    #endif
}

//------------------------------------------------------------------------------------------------------------

// Fused Conv+MaxPool dispatcher: hardware accelerator when CONV_XON, else the C golden model.
void conv_pool_fused(uint8_t* pool_arr_out,                              // pooled output feature-map
                     uint8_t* conv_arr_in,                               // conv input image / feature-map
                     int      arr_in_dim,                                // conv input dimension
                     int8_t*  kernel_w,                                  // 5x5 conv kernel weights
                     int32_t  kernel_b) {                                // conv scalar bias

    #ifdef CONV_XON
    conv_pool_fused_xlr(pool_arr_out, conv_arr_in, arr_in_dim, kernel_w, kernel_b);
    #else
    conv_pool_fused_nox(pool_arr_out, conv_arr_in, arr_in_dim, kernel_w, kernel_b);
    #endif
}




