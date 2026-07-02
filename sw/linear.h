#ifndef _LINEAR_H_
#define _LINEAR_H_


#include <k5_libs.h>
#include <slr_lib.h>
#include "slrx.h" 

//------------------------------------------------------------------------------------------------------------

// Linear Layer element

uint8_t lin_elem(uint8_t* lin_arr_in,   // linear Input Image (single row)
              int         lin_in_dim,   // linear Input dimensions
              int8_t*     linear_w_trn, // linear Weights Transposed, can be negative (2D matrix)
              int32_t*    linear_b,     // linear Bias, can be negative (single row)
              int         lin_out_idx); // output vector element index

// Linear Layer element, software golden (computes one output element into lin_arr_out[lin_out_idx])
void lin_elem_nox(uint8_t* lin_arr_out,  // linear output feature-map (single row)
                  uint8_t* lin_arr_in,   // linear Input Image (single row)
                  int      lin_in_dim,   // linear Input dimensions
                  int8_t*  linear_w_trn, // linear Weights Transposed, can be negative (2D matrix)
                  int32_t* linear_b,     // linear Bias, can be negative (single row)
                  int      lin_out_idx); // output vector element index


//------------------------------------------------------------------------------------------------------------

 // Linear Layer 

void linear(uint8_t* lin_arr_out,  // linear output feature-map (single row)
            uint8_t* lin_arr_in,   // linear Input Image (single row)
            int      lin_in_dim,   // linear Input dimensions
            int      lin_out_dim,  // linear Input dimensions
            int8_t*  linear_w_trn, // linear Weights Transposed, can be negative (2D matrix)
            int32_t* linear_b);    // linear Bias, can be negative (single row)


//------------------------------------------------------------------------------------------------------------

 // Linear Layer with fused hardware Select (argmax) -> writes the winning index byte to sel_result

void linear_sel(uint8_t* lin_arr_out,  // linear output feature-map (single row)
                uint8_t* lin_arr_in,   // linear Input Image (single row)
                int      lin_in_dim,   // linear Input dimensions
                int      lin_out_dim,  // linear Output dimensions
                int8_t*  linear_w_trn, // linear Weights Transposed, can be negative (2D matrix)
                int32_t* linear_b,     // linear Bias, can be negative (single row)
                uint8_t* sel_result,   // XMEM scratch byte for the argmax index
                int      sel_en);      // 1 -> run the fused Select after the layer


//------------------------------------------------------------------------------------------------------------

#endif