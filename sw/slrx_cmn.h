#ifndef _SLRX_CMN_H_
#define _SLRX_CMN_H_

#include <k5_libs.h>
#include <slr_lib.h>

// ===== ACTIVE BUILD CONFIG: verified fused pipeline + B1 (command fusion) =====
//  ON below: CONV_POOL_FUSED, LIN_SELECT_FUSED, LIN_FUSED_SETUP. Build with -ccd1 ALL_XON on the
//  fixed top-level slrx.sv (edge-detected done-clear). LIN_BACKTOBACK + LIN_MACPIPE + S2 tree are
//  source-level ON in hw/linear/linear.sv. Self-checks (CONV_FUSED_SELFCHECK / LIN_SELECT_SELFCHECK)
//  stay OFF for the -stm -itr 1 cycle run; turn them ON for the correctness run
//  (expect "we are the champions" 20/20, LIN1 0/27, P0 0/196, P1 0/25).
// ==============================================================================

#ifdef ALL_XON
#define CONV_XON
#define POOL_XON
#define LIN_XON
#endif

// Fused Conv+MaxPool toggle: uncomment to use the fused path (conv_pool_fused) for
// C0->P0 and C1->P1 instead of the separate conv + pool passes. Default OFF keeps the
// proven separate path as the known-good build / fallback.
//   - CONV_POOL_FUSED alone           -> Stage A: software golden model (conv_pool_fused_nox)
//   - CONV_POOL_FUSED + CONV_XON       -> Stage B: hardware fused accelerator (conv.sv)
// In fused mode the separate POOL accelerator is never invoked, so POOL_XON has no effect.
#define CONV_POOL_FUSED

// Debug: when set (together with CONV_POOL_FUSED + CONV_XON), after the HW fused convs run,
// recompute the golden pooled maps in software and print where the HW diverges. Non-destructive.
// #define CONV_FUSED_SELFCHECK

// Fused hardware Select: when set (requires LIN_XON), Linear1 runs via linear_sel(), and the linear
// accelerator computes the final argmax on-chip and returns the index in an XMEM scratch -> the SW
// get_max_val_idx() and the 27-byte read-back are skipped. Default OFF keeps the SW Select path.
#define LIN_SELECT_FUSED

// Debug: when set (together with LIN_SELECT_FUSED), recompute the golden Linear1 vector + argmax in
// software on image 0 and print any mismatch vs the HW output / index. Non-destructive.
// #define LIN_SELECT_SELFCHECK

// Stage 2B command fusion (requires LIN_XON): fold LIN_SETUP into LIN_CALC so each linear layer issues
// ONE accelerator command + ONE done-poll instead of two. The driver sets OUT_COL_IDX_RI[1]=1 and skips
// the separate LIN_SETUP command; the HW (linear.sv) then reads the input vector + bulk bias at the start
// of the LIN_CALC activation. Pure schedule change (bit-exact). Default OFF = the verified 2-command path.
#define LIN_FUSED_SETUP


//------------------------------------------------------------------------------------------------------------

uint8_t relu_and_descale(int32_t x);

#endif
