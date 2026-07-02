#ifndef _SLRX_H_
#define _SLRX_H_

#include "../../../hw/xlrs/slrx/top/slrx_enums.svh"
#include "slrx_cmn.h" 
#include "conv.h" 
#include "pool.h"
#include "linear.h"

// Register Pointers
#ifndef HLCM
#define HOST_REG(HOST_REG_IDX) (*((volatile unsigned int *)(XBOX_REGS_BASE_ADDR + (4*HOST_REG_IDX))))
#endif

#endif