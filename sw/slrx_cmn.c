#include <k5_libs.h>
#include <slr_lib.h>

//------------------------------------------------------------------------------------------------------------

uint8_t relu_and_descale(int32_t x) {
      
    int32_t ret_val_int = 0 ; // by default per RELU all negative values return 0 
    
    if (x>0) {         
       ret_val_int = x/256 ;
       if (ret_val_int > 255) ret_val_int = 255; // clamp at 255 (max uint8_t)
    }
    return (uint8_t)ret_val_int ;
}


//------------------------------------------------------------------------------------------------------------
