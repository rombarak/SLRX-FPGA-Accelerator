#include <k5_libs.h>
#include <slr_lib.h>
#include "slrx.h" 

//------------------------------------------------------------------------------------------------------------

void lin_elem_setup(uint8_t* lin_arr_out,  // linear output feature-map (single row)
                    uint8_t* lin_arr_in,   // linear Input Image (single row)
                    int      lin_in_dim,   // linear Input dimensions 
                    int      lin_out_dim,   // linear Output dimensions                     
                    int8_t* linear_w_trn, // linear Weights Transposed, can be negative (2D matrix)
                    int32_t* linear_b) {   // linear Bias, can be negative (single row)

    #ifdef HLCM    
    printf("HLCM does not support HW acceleration, quitting\n\n");
    bm_quit_app();
    #else

    HOST_REG(ARR_OUT_ADDR_RI)  = (unsigned int)lin_arr_out;
    HOST_REG(ARR_IN_ADDR_RI)   = (unsigned int)lin_arr_in;
    HOST_REG(ARR_IN_DIM_RI)    = lin_in_dim;
    HOST_REG(ARR_OUT_DIM_RI)   = lin_out_dim;    
    HOST_REG(WGT_ADDR_RI)      = (unsigned int)linear_w_trn;
    HOST_REG(LIN_BIAS_ADDR_RI) = (unsigned int)linear_b;
    #ifndef LIN_FUSED_SETUP
    HOST_REG(XLR_START_RI) = LIN_SETUP ; // SETUP is defined at included ../../../hw/top/slrx_enums.svh

    while (!HOST_REG(XLR_DONE_RI)) {
       //printf("Pool setup Polling ...\n");
    }
    #else
    // Stage 2B command fusion: do NOT issue a separate LIN_SETUP command. The 6 regs above are staged;
    // the upcoming LIN_CALC (with OUT_COL_IDX_RI[1]=1) runs SETUP itself -> one command + one poll/layer.
    #endif
    #endif
}

//------------------------------------------------------------------------------------------------------------

void lin_elem_nox(uint8_t* lin_arr_out,   // linear output feature-map (single row)
                  uint8_t* lin_arr_in,    // linear Input Image (single row)
                  int      lin_in_dim,    // linear Input dimensions           
                  int8_t* linear_w_trn,  // linear Weights Transposed, can be negative (2D matrix)
                  int32_t* linear_b,      // linear Bias, can be negative (single row)
                  int      lin_out_idx) { // output vector element index
         
         int32_t acc = linear_b[lin_out_idx];
         
         for (int lin_in_idx = 0; lin_in_idx < lin_in_dim; lin_in_idx++) {       
             int linear_w_idx = (lin_out_idx * lin_in_dim) + lin_in_idx ;
             acc += (int32_t)(lin_arr_in[lin_in_idx]) * (int32_t)(((volatile int8_t*)linear_w_trn)[linear_w_idx]);
         }
         
         uint8_t lin_elem_out = relu_and_descale(acc); 
         ((volatile uint8_t*)lin_arr_out)[lin_out_idx] = lin_elem_out ;
}

//------------------------------------------------------------------------------------------------------------

 // Linear Layer 

void linear(uint8_t* lin_arr_out,     // linear output feature-map (single row)
            uint8_t* lin_arr_in,      // linear Input Image (single row)
            int      lin_in_dim,      // linear Input dimensions
            int      lin_out_dim,     // linear Input dimensions              
            int8_t* linear_w_trn,    // linear Weights Transposed, can be negative 
            int32_t* linear_b) {      // linear Bias, can be negative (single row)


    #ifdef LIN_XON
    lin_elem_setup(lin_arr_out, lin_arr_in, lin_in_dim, lin_out_dim, linear_w_trn, linear_b);

    #ifdef LIN_FUSED_SETUP
    HOST_REG(OUT_COL_IDX_RI) = 0x2; // sel_en=0 (bit0); bit1=1 -> this LIN_CALC self-runs SETUP first
    #else
    HOST_REG(OUT_COL_IDX_RI) = 0;   // sel_en = 0 : no fused Select for this layer (e.g. Linear0).
    #endif
    // One command runs the whole output vector; the CPU polls once for the entire batch.
    HOST_REG(XLR_START_RI) = LIN_CALC;
    while (!HOST_REG(XLR_DONE_RI)) {
       // HW is running autonomously, polling once for the entire batch...
    }

    #else
    // Plain software path (no HW acceleration): reference per-element loop, unchanged.
    for (int lin_out_idx = 0; lin_out_idx < lin_out_dim; lin_out_idx++) {
        lin_elem_nox(lin_arr_out, lin_arr_in, lin_in_dim, linear_w_trn, linear_b, lin_out_idx);
    }
    #endif
}

//------------------------------------------------------------------------------------------------------------

// Linear Layer with FUSED hardware Select (argmax). Same as linear(), but when sel_en is set the
// accelerator computes the argmax of the output vector on-chip and writes the winning index (1 byte) to
// the XMEM scratch 'sel_result'. The HW also still block-writes the full output vector to lin_arr_out.
// Register reuse: OUT_ROW_IDX_RI carries sel_result's address, OUT_COL_IDX_RI[0] carries sel_en.
void linear_sel(uint8_t* lin_arr_out,  // linear output feature-map (single row)
                uint8_t* lin_arr_in,   // linear Input Image (single row)
                int      lin_in_dim,   // linear Input dimensions
                int      lin_out_dim,  // linear Output dimensions
                int8_t*  linear_w_trn, // linear Weights Transposed, can be negative (2D matrix)
                int32_t* linear_b,     // linear Bias, can be negative (single row)
                uint8_t* sel_result,   // XMEM scratch byte the HW writes the argmax index into
                int      sel_en) {     // 1 -> run the fused Select after the layer (e.g. Linear1)

    #ifdef LIN_XON
    lin_elem_setup(lin_arr_out, lin_arr_in, lin_in_dim, lin_out_dim, linear_w_trn, linear_b);

    HOST_REG(OUT_ROW_IDX_RI) = (unsigned int)sel_result; // where HW writes the index
    #ifdef LIN_FUSED_SETUP
    HOST_REG(OUT_COL_IDX_RI) = (sel_en & 0x1) | 0x2;       // bit0=sel_en; bit1=fused SETUP inside this LIN_CALC
    #else
    HOST_REG(OUT_COL_IDX_RI) = sel_en;                     // enable fused Select
    #endif
    HOST_REG(XLR_START_RI)   = LIN_CALC;
    while (!HOST_REG(XLR_DONE_RI)) {
       // one poll for the whole vector + the fused argmax
    }

    #else
    // SW fallback: compute the vector, then replicate get_max_val_idx (incl. its int8_t quirk) into sel_result.
    for (int lin_out_idx = 0; lin_out_idx < lin_out_dim; lin_out_idx++) {
        lin_elem_nox(lin_arr_out, lin_arr_in, lin_in_dim, linear_w_trn, linear_b, lin_out_idx);
    }
    if (sel_en && sel_result) {
        int    max_idx = 0;
        int8_t max_val = ((volatile uint8_t*)lin_arr_out)[0];
        for (int i = 1; i < lin_out_dim; i++) {
            if (((volatile uint8_t*)lin_arr_out)[i] > max_val) {
                max_idx = i;
                max_val = ((volatile uint8_t*)lin_arr_out)[i];
            }
        }
        ((volatile uint8_t*)sel_result)[0] = (uint8_t)max_idx;
    }
    #endif
}