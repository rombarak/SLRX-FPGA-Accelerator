#ifndef _CONV_H_
#define _CONV_H_

#include <k5_libs.h>
#include <slr_lib.h>
#include "slrx.h" 

//---------------------------------------------------------------------------------------------------------------------------------


void conv(uint8_t* conv_arr_out,                               // Conv output feature-map
          uint8_t* conv_arr_in,                                // Conv Input Image
          int      arr_in_dim,                                 // Conv Input dimensions
          int8_t*  kernel_w, ///  int8_t   kernel_w[CONV_KERNEL_DIM][CONV_KERNEL_DIM], // Conv kernel Weights, can be negative
          int32_t  kernel_b);

//---------------------------------------------------------------------------------------------------------------------------------

// Single 5x5 conv window golden value (relu_and_descale of bias + dot product). Exposed for
// the fused HW self-check so it can print the four windows feeding a pooled cell.
uint8_t conv_window_val_nox(uint8_t* conv_arr_in,                         // Conv Input Image
                            int      arr_in_dim,                          // Conv Input array dimensions
                            int      out_row_idx,                         // output array row index
                            int      out_col_idx,                         // output array column index
                            int8_t*  kernel_w,                            // 5x5 conv kernel weights
                            int32_t  kernel_b);                           // conv scalar bias

// Fused Conv(5x5) + 2x2 MaxPool software golden model: writes the POOLED feature-map
// directly, with no intermediate conv feature-map. Bit-identical to conv() + pool_max_2x2().
void conv_pool_fused_nox(uint8_t* pool_arr_out,                // pooled output feature-map (post_pX_fm)
                         uint8_t* conv_arr_in,                 // conv input image / feature-map
                         int      arr_in_dim,                  // conv input dimension
                         int8_t*  kernel_w,                    // 5x5 conv kernel weights
                         int32_t  kernel_b);                   // conv scalar bias

// Fused Conv+MaxPool dispatcher: hardware accelerator when CONV_XON, else the golden model.
void conv_pool_fused(uint8_t* pool_arr_out,                    // pooled output feature-map (post_pX_fm)
                     uint8_t* conv_arr_in,                     // conv input image / feature-map
                     int      arr_in_dim,                      // conv input dimension
                     int8_t*  kernel_w,                        // 5x5 conv kernel weights
                     int32_t  kernel_b);                       // conv scalar bias


#endif


