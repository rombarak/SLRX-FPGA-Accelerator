#include <k5_libs.h>
#include <slr_lib.h>
#include "slrx.h" 

//------------------------------------------------------------------------------------------------------------

// Get index of max value in vector

int get_max_val_idx(uint8_t*  vec_in, int vec_size) { // input vector
  
   int max_val_idx = 0 ;
   int8_t max_val = ((volatile uint8_t*)vec_in)[0] ;
   
   for (int i=1;i<vec_size;i++) {
     if (vec_in[i]>max_val) {
        max_val_idx = i ;
        max_val = ((volatile uint8_t*)vec_in)[i] ;
     }
   }
   return max_val_idx ;
}

//------------------------------------------------------------------------------------------------------------

#ifdef LIN_SELECT_FUSED
// XMEM scratch where the linear accelerator writes the fused-Select argmax index (alloc'd once in main()).
static uint8_t* g_lin_sel_idx = NULL;
#endif

//------------------------------------------------------------------------------------------------------------

#ifdef CONV_FUSED_SELFCHECK
// Compare a hardware-produced pooled feature-map against the software golden model, byte by
// byte, and report the divergence (count + first few mismatches with values). Used to
// localize a bug in the fused conv hardware. Read-only w.r.t. the hw buffer.
static void conv_fused_selfcheck(const char* tag, uint8_t* hw, uint8_t* golden, int dim) {
  int n = dim * dim;
  int mism = 0;
  for (int i = 0; i < n; i++) {
    if (hw[i] != golden[i]) {
      if (mism < 16) printf("  %s mismatch idx=%d (row=%d col=%d): hw=%d golden=%d\n",
                            tag, i, i / dim, i % dim, hw[i], golden[i]);
      mism++;
    }
  }
  printf("  %s: %d / %d mismatches\n", tag, mism, n);
}
#endif

//============================================================================================================

int infer(uint8_t slr_img [IMG_DIM][IMG_DIM],      // inferred image
          slr_model_params_t* slr_model_params_p,  // model parameters
          slr_intr_fm_t*      slr_intr_fm_p,       // intermediate feature-maps
          char                check_performance,   // enable performance monitoring 
          int                 img_idx) {           // inferred image index (just for reporting)  
          
     
        if (check_performance) reset_report_performance(); 
     
        // Perform Inference

        #ifdef CONV_POOL_FUSED

        // Fused Conv0 + Pool0 : computes conv on the fly and writes the POOLED map directly,
        // skipping the intermediate post_c0_fm (no separate Pool0 pass). Dispatcher picks
        // the HW accelerator (CONV_XON) or the C golden model.
        conv_pool_fused((uint8_t*)(slr_intr_fm_p->post_p0_fm), // Pool0 output feature-map (fused)
                        (uint8_t*) slr_img,                    // Conv0 Input Image
                                   IMG_DIM,                    // Conv0 Input dimensions
                        (int8_t*) (slr_model_params_p->conv0_w),// Conv0 kernel Weights
                        (int32_t) (slr_model_params_p->conv0_b));// Conv0 kernel Bias

        if (check_performance) report_task_performance("ConvPool0");

        // Fused Conv1 + Pool1 : writes the POOLED map directly, skipping post_c1_fm.
        conv_pool_fused((uint8_t*)(slr_intr_fm_p->post_p1_fm), // Pool1 output feature-map (fused)
                        (uint8_t*)(slr_intr_fm_p->post_p0_fm), // Conv1 Input Image <- output of pool0
                                   POST_P0_FM_DIM,             // Conv1 Input dimensions
                        (int8_t*) (slr_model_params_p->conv1_w),// Conv1 kernel Weights
                        (int32_t) (slr_model_params_p->conv1_b));// Conv1 kernel Bias

        if (check_performance) report_task_performance("ConvPool1");

        #ifdef CONV_FUSED_SELFCHECK
        if (img_idx == 0) {  // one clean dump is enough; the bug is systematic
          // Recompute the golden pooled maps in software and compare to the HW results.
          // post_c0_fm / post_c1_fm are unused in fused mode -> reuse them as golden scratch.
          // P0 check: conv0 HW vs golden(slr_img). P1 check: conv1 HW vs golden(HW post_p0_fm),
          // so conv1 is validated on its real input even if conv0 differs.
          printf("CONV_FUSED_SELFCHECK (img 0):\n");
          conv_pool_fused_nox((uint8_t*)(slr_intr_fm_p->post_c0_fm),  // golden P0 scratch
                              (uint8_t*) slr_img, IMG_DIM,
                              (int8_t*) (slr_model_params_p->conv0_w),
                              (int32_t) (slr_model_params_p->conv0_b));
          conv_fused_selfcheck("P0",
                               (uint8_t*)(slr_intr_fm_p->post_p0_fm),
                               (uint8_t*)(slr_intr_fm_p->post_c0_fm),
                               POST_P0_FM_DIM);

          conv_pool_fused_nox((uint8_t*)(slr_intr_fm_p->post_c1_fm),  // golden P1 scratch
                              (uint8_t*)(slr_intr_fm_p->post_p0_fm), POST_P0_FM_DIM,
                              (int8_t*) (slr_model_params_p->conv1_w),
                              (int32_t) (slr_model_params_p->conv1_b));
          conv_fused_selfcheck("P1",
                               (uint8_t*)(slr_intr_fm_p->post_p1_fm),
                               (uint8_t*)(slr_intr_fm_p->post_c1_fm),
                               POST_P1_FM_DIM);
        }
        #endif

        #else

        // Calling CONV0 Layer
        conv((uint8_t*)(slr_intr_fm_p->post_c0_fm),         // Conv0 output feature-map
             (uint8_t*) slr_img,                            // Conv0 Input Image
                        IMG_DIM,                            // Conv0 Input dimensions
             (int8_t*) (slr_model_params_p->conv0_w),       // Conv0 kernel Weights
             (int32_t) (slr_model_params_p->conv0_b));      // Conv0 kernel Bias

        if (check_performance) report_task_performance("Conv0");

        // Calling POOL0 Layer
        pool_max_2x2((uint8_t*)(slr_intr_fm_p->post_p0_fm), // Pool0 output feature-map
                     (uint8_t*)(slr_intr_fm_p->post_c0_fm), // Pool0 Input Image <- output of conv0
                     POST_C0_FM_DIM);                       // Pool0 Input dimensions <- post conv0

        if (check_performance) report_task_performance("Pool0");

        // Calling CONV1 Layer
        conv((uint8_t*)(slr_intr_fm_p->post_c1_fm),         // Conv1 output feature-map <- Pool0 output feature-map
             (uint8_t*)(slr_intr_fm_p->post_p0_fm),         // Conv1 Input Image
                        POST_P0_FM_DIM,                     // Conv1 Input dimensions
             (int8_t*) (slr_model_params_p->conv1_w),       // Conv1 kernel Weights
             (int32_t) (slr_model_params_p->conv1_b));      // Conv1 kernel Bias


        if (check_performance) report_task_performance("Conv1");

        // Calling POOL1 Layer
        pool_max_2x2((uint8_t*)(slr_intr_fm_p->post_p1_fm), // Pool1 output feature-map
                     (uint8_t*)(slr_intr_fm_p->post_c1_fm), // Pool1 Input Image <- output of conv1
                     POST_C1_FM_DIM);                       // Pool1 Input dimensions <- post conv1

        if (check_performance) report_task_performance("Pool1");

        #endif // CONV_POOL_FUSED

        // Calling LINEAR0 Layer
        linear((uint8_t*)(slr_intr_fm_p->post_lin0_fm),     // linear output feature-map (single row) 
               (uint8_t*)(slr_intr_fm_p->post_p1_fm),       // linear Input Image (single row) <- output of pool1 (flat)
               LIN_INVEC_SIZE,                              // linear Input dimension
               LIN_HID_DIM,                                 // linear Output dimension              
               (int8_t*)(slr_model_params_p->lin0_w_trn),   // linear Weights Transposed (2D matrix)
               slr_model_params_p->lin0_b);                 // linear Bias (single row)

        if (check_performance) report_task_performance("Linear0");  
        
        // Calling LINEAR1 Layer (+ fused hardware Select when LIN_SELECT_FUSED)
        #ifdef LIN_SELECT_FUSED
        // The linear accelerator computes Linear1 AND the argmax on-chip and writes the winning index
        // (1 byte) to an XMEM scratch -> no separate SW Select pass and no 27-byte read-back.
        linear_sel((uint8_t*)(slr_intr_fm_p->post_lin1_fm),
                   (uint8_t*)(slr_intr_fm_p->post_lin0_fm),
                   LIN_HID_DIM, NUM_LABELS,
                   (int8_t*)(slr_model_params_p->lin1_w_trn),
                   slr_model_params_p->lin1_b,
                   g_lin_sel_idx, 1);

        if (check_performance) report_task_performance("Linear1");

        int detected_label_idx = ((volatile uint8_t*)g_lin_sel_idx)[0];
        #else
        linear((uint8_t*)(slr_intr_fm_p->post_lin1_fm),     // linear output feature-map (single row)
               (uint8_t*)(slr_intr_fm_p->post_lin0_fm),     // linear Input Image (single row) <- output of pool1 (flat)
               LIN_HID_DIM,                                 // linear Input dimension
               NUM_LABELS,                                  // linear Output dimensions
               (int8_t*)(slr_model_params_p->lin1_w_trn),   // linear Weights Transposed (2D matrix)
               slr_model_params_p->lin1_b);                 // linear Bias (single row)

        if (check_performance) report_task_performance("Linear1");

        // Final label selection - Performed on LINEAR1 output vector
        int detected_label_idx = get_max_val_idx((uint8_t*)(slr_intr_fm_p->post_lin1_fm), NUM_LABELS);
        #endif

        if (check_performance) {
          report_task_performance("Select");
          report_total_performance();
          printf("\n");
        }

        #ifdef LIN_SELECT_SELFCHECK
        if (img_idx == 0) {  // verify the HW Linear1 vector + fused argmax vs the SW golden (one clean dump)
          uint8_t lin1_golden[NUM_LABELS];
          for (int i = 0; i < NUM_LABELS; i++)
            lin_elem_nox(lin1_golden, (uint8_t*)(slr_intr_fm_p->post_lin0_fm), LIN_HID_DIM,
                         (int8_t*)(slr_model_params_p->lin1_w_trn), slr_model_params_p->lin1_b, i);
          int g_idx = get_max_val_idx(lin1_golden, NUM_LABELS);
          int mism = 0;
          for (int i = 0; i < NUM_LABELS; i++) {
            uint8_t hw = ((volatile uint8_t*)(slr_intr_fm_p->post_lin1_fm))[i];
            if (hw != lin1_golden[i]) {
              if (mism < 16) printf("  LIN1 mismatch idx=%d: hw=%d golden=%d\n", i, hw, lin1_golden[i]);
              mism++;
            }
          }
          printf("  LIN1: %d / %d mismatches; Select hw=%d golden=%d %s\n",
                 mism, NUM_LABELS, detected_label_idx, g_idx,
                 (detected_label_idx == g_idx) ? "OK" : "MISMATCH");
        }
        #endif

        return detected_label_idx ;
}   

//====================================================================================================

int main() {

  bm_printf("\n\nHELLO K5X SLR : Sign Language Recognition\n\n"); 
  
  alloc_init(); // Initialize XMEM memory allocator
  
  slr_model_params_t* slr_model_params_p = load_model_params();
  
  // Allocating Layers Intermediate Feature-Maps     
  int fm_total_num_bytes = sizeof(slr_intr_fm_t); 
  printf("Allocated total %d bytes of for intermediate feature -maps\n", fm_total_num_bytes);
  slr_intr_fm_t* slr_intr_fm_p = (slr_intr_fm_t*)alloc_get(fm_total_num_bytes, "slr_intr_fm");

  #ifdef LIN_SELECT_FUSED
  g_lin_sel_idx = (uint8_t*)alloc_get(4, "lin_sel_idx"); // XMEM scratch for the fused-Select argmax index
  #endif
   
  int num_imgs_in_buf = 1 ; // Currently we load a single image per iteration
  int imgs_total_num_bytes = num_imgs_in_buf * sizeof(slr_ds_image_t);
  
  slr_ds_image_t* slr_ds_imgs = (slr_ds_image_t*)alloc_get(imgs_total_num_bytes, "slr_ds_imgs");
  
  // Demo/checking guide Modification #1: use the runtime-provided dataset path (needed for
  // animation + LST speed-test, and harmless otherwise). Requires the updated shared env (git pull).
  // char* ds_test_file_path = "$SLR_SHARED/pt/workspace/slr_ds_mnx.txt";  // (original, replaced)
  char ds_test_file_path[40] ;
  get_ds_test_file_path(ds_test_file_path) ;

  FILE_REF file_ref ; // assigned by load_hex_file( ... OPEN*)
  
  printf("\n\n");
  
  for (int img_idx=0; img_idx<NUM_TEST_IMAGES; img_idx++) {

    // Demo/checking guide Modification #2: run-time animation hook + LST (loop speed test) entry.
    // Both are no-ops in a normal run; LST (with -ccd2 LST) takes over and breaks the loop.
    slr_animate_hook(ds_test_file_path) ;
    if (loop_speed_test_on((infer_cb_t)infer, slr_ds_imgs, slr_model_params_p, slr_intr_fm_p)) break ;

    file_access_mode_t file_access_mode = img_idx==0 ? OPEN : CONT ;
    load_hex_file(ds_test_file_path, &file_ref, (char*)slr_ds_imgs, imgs_total_num_bytes, file_access_mode); // Keep Open
  
    char check_performance = (NUM_TEST_IMAGES==1); // check performance (tested on a single image  invocation, for now)    
    char is_last_ds_img    = slr_ds_imgs[0].is_last_img || (NUM_TEST_IMAGES==1) ;
    int  expected_label_id  = slr_ds_imgs[0].slr_img_label_id ; // Currently the buffer holds just one image at a time
       
    int detected_label_id = infer(slr_ds_imgs[0].slr_img, // inferred image 
                                  slr_model_params_p,     // model parameters
                                  slr_intr_fm_p,          // interm feature-maps memory space
                                  check_performance,      // enable performance check
                                  img_idx);               // running image index
          
    char text_mode = TRUE; // oppose to dataset bench-marking
    output_detection(detected_label_id, expected_label_id, img_idx, is_last_ds_img, text_mode);    

    if (is_last_ds_img || ((img_idx+1)>=NUM_TEST_IMAGES)) break ; 
  }
 
  fclose(file_ref);
     
  // Free all allocated memory space 
  
  alloc_free((void*)slr_model_params_p, "model_params");     
  alloc_free((void*)slr_ds_imgs,        "slr_ds_imgs"); 
  alloc_free((void*)slr_intr_fm_p,      "slr_intr_fm"); 
  
  if (NUM_TEST_IMAGES==1) printf("\n\nDone Single Detection With Performance Check\n");

  bm_quit_app();
  return 0;

}




