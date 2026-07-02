//==========================================================================================================
//  slrx_enums.svh -- SHARED CPU <-> accelerator interface contract.
//  Included by BOTH the C drivers (sw/*.c) and the SystemVerilog accelerators (hw/**/*.sv), so ONLY
//  typedef syntax common to C and SystemVerilog is allowed here. The enum POSITION is the register index /
//  command id -- do NOT reorder or rename any member (it is the wire/ABI contract).
//
//  Handshake: the CPU writes the address/dim registers, then writes a slrx_cmd_t value to XLR_START_RI;
//  the addressed accelerator runs (ours are hardware-looped: one command = one whole layer), raises
//  XLR_DONE_RI and holds it until read; the CPU polls `while (!HOST_REG(XLR_DONE_RI)) {}`.
//
//  slr_xlr_t            accelerator ids : CONV=0, POOL=1, LIN=2.
//  conv_host_regs_idx_t host registers :
//     [0]  XLR_START_RI      write a command here -> starts the addressed accelerator
//     [1]  XLR_DONE_RI       poll target: nonzero when the addressed accelerator is done
//     [2]  WGT_ADDR_RI       conv 5x5 kernel (25 int8) OR linear weight matrix base address
//     [3]  LIN_BIAS_ADDR_RI  linear bias vector (int32[N]) base address
//     [4]  CONV_BIAS_VAL_RI  scalar conv bias (int32) -- a VALUE, not an address
//     [5]  ARR_IN_ADDR_RI    input  feature-map / vector base address
//     [6]  ARR_OUT_ADDR_RI   output feature-map / vector base address
//     [7]  ARR_IN_DIM_RI     input  dimension (row length / input-vector length)
//     [8]  ARR_OUT_DIM_RI    output dimension (linear output length)
//     [9]  OUT_ROW_IDX_RI    pool: output row idx | fused linear-Select: XMEM scratch addr for the index byte
//     [10] OUT_COL_IDX_RI    fused linear bitfield: bit0=sel_en, bit1=fused-SETUP, bit2=input-resident
//  slrx_cmd_t           commands :
//     [0] CONV_SETUP   conv latch-only no-op (kept for symmetry; the fused path does not need it)
//     [1] CONV_WINDOW  conv: run the WHOLE fused Conv+MaxPool layer (hardware-looped)
//     [2] POOL_SETUP / [3] POOL_CALC   reference pool (unused in fused mode)
//     [4] LIN_SETUP   linear: read the input vector once + bulk-read the whole bias vector
//     [5] LIN_CALC    linear: stream weight rows, MAC, accumulate, block-write (+ fused argmax if sel_en)
//==========================================================================================================
// Register indexes
// Notice this '.h' file is also included by the accelerate verilog code 
// This only verilog and C common typedef syntax is allowed. 

typedef enum  {
    CONV,              
    POOL,
    LIN,
    NUM_XLRS // Just to indicate number of accelerators
} slr_xlr_t ;


typedef enum  {
    XLR_START_RI,
    XLR_DONE_RI,              
    WGT_ADDR_RI,            // Weights of either Convolution kernel or linear
    LIN_BIAS_ADDR_RI,
    CONV_BIAS_VAL_RI,    
    ARR_IN_ADDR_RI, 
    ARR_OUT_ADDR_RI,
    ARR_IN_DIM_RI,  
    ARR_OUT_DIM_RI,      
    OUT_ROW_IDX_RI,  
    OUT_COL_IDX_RI            
}   conv_host_regs_idx_t;


typedef enum  {
    CONV_SETUP,              
    CONV_WINDOW,
    POOL_SETUP,              
    POOL_CALC,
    LIN_SETUP, 
    LIN_CALC,    
    NUM_SLRX_CMDS  // Just to indicate max index of commands
}   slrx_cmd_t;
