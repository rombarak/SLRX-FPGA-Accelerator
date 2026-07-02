import xbox_def_pkg::*;
import slrx_def_pkg::*;

//--------------------------------------------------------------------------------------------------------
// Fused Convolution(5x5, no padding) + 2x2 MaxPool accelerator, HARDWARE-LOOPED.
// ===== ACTIVE KEEPER (2026-06-28) ===== Stage C step 3 (4-MAC 4-stage pipeline) + PREFETCH:
// the next pooled row's input rows are READ DURING the current row's compute, so the per-row read
// latency is hidden. ONE pooled cell issued per cycle; the per-row LOAD is eliminated (LOAD runs
// only once, for the pr=0 initial fill).
//
// WHY THIS IS THE KEEPER (the earlier "hang" was NOT this RTL): in the `-stm -itr 1` measurement this
// design previously appeared to HANG. ROOT CAUSE was the TOP-LEVEL `slrx.sv` DONE-handshake bug (DONE
// was cleared on ANY read pulse, so the measurement poll never latched it) -- NOT a muxed-port
// read-during-CALC deadlock. The lecturer's FIXED `slrx.sv` (edge-detected `clear_done_on_read`,
// saved at hw/slrx.sv) cures it; with that top-level deployed the prefetch COMPLETES and is the
// fastest conv.  >>> THIS DESIGN REQUIRES the fixed top-level `slrx.sv`. <<<
// (Was conv.sv.prefetch_hang.bak -- identical RTL; the "_hang" name is historical only.)
//
// A single CONV_WINDOW command computes a whole layer and writes the POOLED feature-map directly.
// Per pooled cell (pr,pc) the four 5x5 conv windows of the 2x2 block are:
//     win0=(2pr,  2pc)   win1=(2pr,  2pc+1)     <- top    conv row (wr=0)
//     win2=(2pr+1,2pc)   win3=(2pr+1,2pc+1)     <- bottom conv row (wr=1)
// FOUR parallel MAC windows compute all four conv windows of a pooled cell; ONE pooled cell is
// issued per cycle (4-stage pipeline; see Stage C step 3 below).
//
// COMPUTE PIPELINE (unchanged from the verified Stage C step 3 -> bit-exact):
//   S0 SELECT : mux the four windows' 5x5 input bytes out of the line buffer        -> win_buf (reg)
//   S1 MULT   : 4 windows x (25 products -> 5 row partial sums) from win_buf+kernel  -> gpart   (reg)
//   S2 COMBINE: 4 x (sum the 5 partials + bias, then relu_descale)                   -> act     (reg)
//   S3 MAXPOOL: 2x2 max of the four activations                                      -> out_row_buf
// A 3-deep {valid,col} tag pipeline (s0/s1/s2) names the cell in each stage and drains the tail.
//
// INCREMENT 1 -- PREFETCH (cycle reduction, ~zero Fmax cost):
//   The line buffer is 8 rows (slots 0..5 = the 6 rows this pooled row computes on; slots 6,7 = the
//   2 NEW rows the NEXT pooled row needs). While CALC computes pooled row r (read port otherwise
//   idle), a small prefetch sub-FSM (pf_phase/pf_slot/pf_done) READS input rows 2r+6, 2r+7 into
//   slots 6,7. At WRITE the buffer rolls up by 2 reusing those prefetched rows
//   (in_buf[0..3]<=[2..5], [4..5]<=[6..7]) and goes STRAIGHT BACK TO CALC -- there is NO per-row
//   LOAD anymore. LOAD now runs only ONCE, for the pr=0 initial 6-row fill. This hides ~2x the
//   per-row read latency behind the compute, which is the bulk of the old ConvPool0 overhead.
//   It is pure read/compute overlap (no longer combinational paths), so Fmax is unaffected, and the
//   arithmetic functions are untouched -> still bit-exact vs conv_pool_fused_nox().
//
// Known-good fallbacks: conv.sv.stageC3_4mac_verified.bak (this design WITHOUT prefetch, cloud-proven
// 8,060 LE @ 72.20 MHz), conv.sv.stageC1.bak (2-MAC), conv_stageB.sv.bak (single MAC).
//--------------------------------------------------------------------------------------------------------

// item 5 (back-to-back initial LOAD): hold mem_req high and pipeline the pr=0 six-row fill via a
// latency-1 lagging-index bind (the SAME proven trick as linear's LIN_BACKTOBACK). Only the initial
// fill changes; the prefetch reads are already hidden behind compute, so they are left untouched.
// Comment this line to revert to the per-row request+settle LOAD (the verified 927 keeper).
`define CONV_LOAD_B2B

// item 4 (write/compute overlap): snapshot the finished pooled row into out_wbuf and flush it to XMEM
// via a write-engine that runs DURING the NEXT row's CALC. The write is serialized AFTER pf_done, so it
// never co-issues with a prefetch read -> the memory pattern stays "write-during-compute" only (the same
// pattern as the original item-4, which we now believe hung only on the old top-level DONE bug, since
// fixed). Comment this line to revert to the blocking WRITE state (the verified 917 keeper).
`define CONV_WRITE_OVERLAP

module conv (
  input   clk,
  input   rst_n,

  // Command Status Register Interface
  slrx_regs_intrf.xlr slrx_regs_intrf,           // Host Registers Interface

  // muxed interfaces
  mem_intf_read.client_read   mem_intf_read,     // shared XMEM read  port
  mem_intf_write.client_write mem_intf_write     // shared XMEM write port
);

  // ---- dimensions / sizes ----
  localparam DIM_MAX      = 32;                   // all feature-map dims <= 32
  localparam KDIM         = 5;                    // 5x5 kernel
  localparam KSIZE        = KDIM*KDIM;            // 25 kernel taps
  localparam NROWS        = 6;                    // input rows USED to compute one pooled output row
  localparam NPREF        = 2;                    // prefetch slots (the 2 new rows for the next pooled row)
  localparam BUF_ROWS     = NROWS + NPREF;        // 8 = line buffer depth (6 compute + 2 prefetch)
  localparam POOL_OUT_MAX = (DIM_MAX-KDIM+1)/2;   // 14 = max pooled row length
  localparam ACC_W        = 32;                   // dot-product accumulator (matches int32 golden)
  localparam ARR_IDX_W    = $clog2(DIM_MAX);      // 5

  // ---- FSM ----
  enum {IDLE, READ_KERNEL, LOAD, CALC, WRITE, FINAL_WRITE, DONE} next_state, state;

  // ---- host-reg decoded inputs ----
  slrx_cmd_t slrx_cmd;
  logic conv_active, conv_start, conv_done, clear_done_on_read;

  logic [XMEM_ADDR_WIDTH-1:0] conv_kernel_addr;
  logic [XMEM_ADDR_WIDTH-1:0] conv_arr_in_addr;
  logic [XMEM_ADDR_WIDTH-1:0] conv_arr_out_addr;
  logic [ARR_IDX_W:0]         conv_arr_in_dim;
  logic [ARR_IDX_W:0]         conv_out_dim;       // conv feature-map dim = in_dim-5+1
  logic [ARR_IDX_W:0]         pool_out_dim;       // pooled output dim    = conv_out_dim/2
  logic signed [ACC_W-1:0]    conv_bias_s;        // scalar conv bias (int32), combinational from host reg

  // ---- on-chip buffers ----
  logic [KSIZE-1:0][7:0]               kernel_buf,  kernel_buf_ps;   // 25 int8 weights (stationary)
  logic [BUF_ROWS-1:0][DIM_MAX-1:0][7:0] in_buf,    in_buf_ps;       // 8 input rows (6 compute + 2 prefetch)
  logic [POOL_OUT_MAX-1:0][7:0]        out_row_buf, out_row_buf_ps;  // one pooled output row (compute target)
  logic signed [ACC_W-1:0]             bias_reg,    bias_reg_ps;     // latched conv bias (stationary per layer)

  // ---- item 4: write/compute overlap (snapshot the finished row, flush it during the next CALC) ----
  logic [POOL_OUT_MAX-1:0][7:0] out_wbuf, out_wbuf_ps;  // snapshot of a finished row, being written back
  logic               pw_valid, pw_valid_ps;            // out_wbuf holds a row pending write
  logic [ARR_IDX_W:0] pw_pr,    pw_pr_ps;               // pooled-row index the pending row belongs to

  // ---- four MAC windows, 4-stage pipeline (window w = (wr,wc) : 0:(0,0) 1:(0,1) 2:(1,0) 3:(1,1)) ----
  logic [3:0][KSIZE-1:0][7:0]      win_buf, win_buf_ps;  // S0->S1 : the 4 windows' 5x5 input bytes (registered MAC operands)
  logic [3:0][KDIM-1:0][ACC_W-1:0] gpart,   gpart_ps;    // S1->S2 : 4 windows x 5 row partial sums  (registered MAC outputs)
  logic [3:0][7:0]                 act,     act_ps;       // S2->S3 : 4 window activations            (registered)

  // ---- counters / streaming control ----
  logic [ARR_IDX_W:0] pr, pr_ps;              // pooled output row index
  logic [2:0]         load_row, load_row_ps;  // line-buffer slot being loaded in the pr=0 initial LOAD (0..NROWS-1)
  logic               rd_phase, rd_phase_ps;  // initial-LOAD read sub-cycle (0: request+capture, 1: settle)

  logic [ARR_IDX_W:0] issue_col, issue_col_ps; // pooled column being SELECTED into the pipeline (S0)
  // 3-deep tag pipeline : {valid,col} of the cell residing in win_buf (s0), gpart (s1), act (s2)
  logic [ARR_IDX_W:0] s0_col, s0_col_ps,  s1_col, s1_col_ps,  s2_col, s2_col_ps;
  logic               s0_valid, s0_valid_ps, s1_valid, s1_valid_ps, s2_valid, s2_valid_ps;
  logic               comp_done, comp_done_ps;  // this pooled row's last cell has been stored

  // ---- prefetch sub-FSM (reads the 2 new rows for the NEXT pooled row into slots 6,7 during CALC) ----
  logic pf_phase, pf_phase_ps;   // prefetch read sub-cycle (0: request+capture, 1: settle)
  logic pf_slot,  pf_slot_ps;    // which prefetch row: 0 -> slot NROWS (6), 1 -> slot NROWS+1 (7)
  logic pf_done,  pf_done_ps;    // both prefetch rows captured (or none needed, last pooled row)

  // ---- item 5: back-to-back initial LOAD (held-high reads, latency-1 lagging bind) ----
  logic [2:0] ld_fetch,    ld_fetch_ps;     // next slot to REQUEST (0..NROWS)
  logic [2:0] ld_inflight, ld_inflight_ps;  // slot whose data arrives on the NEXT mem_valid (lagging)
  logic       ld_req,      ld_req_ps;       // a request is outstanding -> gate the capture

  // ---- combinational addresses ----
  logic [XMEM_ADDR_WIDTH-1:0] cur_in_row_addr, cur_out_row_addr, pf_row_addr;

  //--------------------------------------------------------------------------------------------------------
  // Host Regs Interface

  assign slrx_regs_intrf.xlr_done = conv_done;

  assign slrx_cmd           = slrx_cmd_t'(slrx_regs_intrf.host_regs[XLR_START_RI][$clog2(NUM_SLRX_CMDS)-1:0]);
  assign conv_active        = (slrx_cmd==CONV_SETUP) || (slrx_cmd==CONV_WINDOW);
  assign conv_start         = slrx_regs_intrf.host_regs_valid_pulse[XLR_START_RI] && conv_active;
  assign clear_done_on_read = conv_active && slrx_regs_intrf.xlr_done_ack;

  assign conv_kernel_addr   = slrx_regs_intrf.host_regs[WGT_ADDR_RI];     // 5x5 kernel weights base
  assign conv_arr_in_addr   = slrx_regs_intrf.host_regs[ARR_IN_ADDR_RI];  // input feature-map base
  assign conv_arr_out_addr  = slrx_regs_intrf.host_regs[ARR_OUT_ADDR_RI]; // pooled output base
  assign conv_arr_in_dim    = slrx_regs_intrf.host_regs[ARR_IN_DIM_RI];   // input array dimension
  assign conv_bias_s        = $signed(slrx_regs_intrf.host_regs[CONV_BIAS_VAL_RI]); // scalar bias

  assign conv_out_dim       = conv_arr_in_dim - KDIM + 1;
  assign pool_out_dim       = conv_out_dim >> 1;                          // /2 (conv dim is even here)

  // pr=0 initial LOAD: input row in slot 'load_row' = 2*pr + load_row (pr=0 here) ; output row written = pr
  assign cur_in_row_addr    = conv_arr_in_addr  + (((pr<<1) + load_row) * conv_arr_in_dim);
  assign cur_out_row_addr   = conv_arr_out_addr + (pr * pool_out_dim);
  // prefetch: the 2 NEW rows the next pooled row (pr+1) needs are input rows 2*pr+6 and 2*pr+7
  assign pf_row_addr        = conv_arr_in_addr  + (((pr<<1) + NROWS + pf_slot) * conv_arr_in_dim);
  // item 5: address of the slot being REQUESTED this cycle in the back-to-back initial LOAD (pr=0)
  logic [XMEM_ADDR_WIDTH-1:0] ld_fetch_addr;
  assign ld_fetch_addr      = conv_arr_in_addr  + (((pr<<1) + ld_fetch) * conv_arr_in_dim);
  // item 4: output address of the row currently pending write (out_wbuf belongs to pooled row pw_pr)
  logic [XMEM_ADDR_WIDTH-1:0] pw_out_addr;
  assign pw_out_addr        = conv_arr_out_addr + (pw_pr * pool_out_dim);

  //========================================================================================================
  // State machine + datapath (combinational)
  always_comb begin

    // ---- defaults ----
    next_state = state;

    mem_intf_read.mem_req         = 0;
    mem_intf_read.mem_start_addr  = 0;
    mem_intf_read.mem_size_bytes  = 0;

    mem_intf_write.mem_req        = 0;
    mem_intf_write.mem_start_addr = cur_out_row_addr;
    mem_intf_write.mem_size_bytes = pool_out_dim;
    mem_intf_write.mem_data       = out_row_buf;

    conv_done   = 0;

    pr_ps        = pr;
    load_row_ps  = load_row;
    rd_phase_ps  = rd_phase;
    issue_col_ps = issue_col;
    s0_col_ps    = s0_col;   s0_valid_ps = s0_valid;
    s1_col_ps    = s1_col;   s1_valid_ps = s1_valid;
    s2_col_ps    = s2_col;   s2_valid_ps = s2_valid;
    comp_done_ps = comp_done;
    pf_phase_ps  = pf_phase;
    pf_slot_ps   = pf_slot;
    pf_done_ps   = pf_done;
    ld_fetch_ps    = ld_fetch;
    ld_inflight_ps = ld_inflight;
    ld_req_ps      = ld_req;
    out_wbuf_ps    = out_wbuf;
    pw_valid_ps    = pw_valid;
    pw_pr_ps       = pw_pr;

    kernel_buf_ps  = kernel_buf;
    in_buf_ps      = in_buf;
    out_row_buf_ps = out_row_buf;
    bias_reg_ps    = bias_reg;
    win_buf_ps     = win_buf;
    gpart_ps       = gpart;
    act_ps         = act;

    case (state)

      IDLE: begin
        if (conv_start) begin
          pr_ps=0; load_row_ps=0; rd_phase_ps=0;
          bias_reg_ps = conv_bias_s;                            // latch the scalar bias for the whole layer
          if (slrx_cmd==CONV_WINDOW) next_state = READ_KERNEL;  // run the whole fused layer
          else                       next_state = DONE;         // CONV_SETUP: latch-only no-op
        end
      end

      READ_KERNEL: begin
        mem_intf_read.mem_req        = 1;
        mem_intf_read.mem_start_addr = conv_kernel_addr;
        mem_intf_read.mem_size_bytes = KSIZE;
        if (mem_intf_read.mem_valid) begin
          integer i;
          for (i=0;i<KSIZE;i++) kernel_buf_ps[i] = mem_intf_read.mem_data[i];
          mem_intf_read.mem_req = 0;
          load_row_ps = 0;            // first pooled row: full 6-row initial load (slots 0..5)
          rd_phase_ps = 0;
          ld_fetch_ps = 0; ld_inflight_ps = 0; ld_req_ps = 0;  // item 5: arm the back-to-back fill
          pw_valid_ps = 0; pw_pr_ps = 0;                       // item 4: arm the write-snapshot per layer
          next_state  = LOAD;
        end
      end

      // ---- LOAD : pr=0 ONLY -- the initial 6-row fill (slots 0..5). For pr>=1 the rows are already
      //      in place from the WRITE roll + prefetch, so WRITE goes straight to CALC (no LOAD). ----
      LOAD: begin
`ifdef CONV_LOAD_B2B
        // item 5 -- BACK-TO-BACK fill: hold mem_req high and present a NEW slot address every cycle.
        // The data on a mem_valid cycle belongs to the slot requested ONE cycle EARLIER (read latency
        // 1), so bind it to the LAGGING ld_inflight -- NOT to ld_fetch (already the next addr on the
        // bus). This is the off-by-one fix proven on the same muxed port by linear's LIN_BACKTOBACK.
        if (ld_fetch < NROWS) begin
          mem_intf_read.mem_req        = 1;
          mem_intf_read.mem_start_addr = ld_fetch_addr;
          mem_intf_read.mem_size_bytes = conv_arr_in_dim;
        end
        // capture whatever returns this cycle into the in-flight slot (gated by ld_req so a valid
        // before any real request is ignored)
        if (mem_intf_read.mem_valid && ld_req) begin
          integer i;
          for (i=0;i<DIM_MAX;i++)
            in_buf_ps[ld_inflight][i] = (i<conv_arr_in_dim) ? mem_intf_read.mem_data[i] : 8'd0;
          if (ld_inflight == NROWS-1) begin          // last slot (5) captured -> start streaming
            issue_col_ps = 0;
            s0_valid_ps  = 1'b0; s1_valid_ps = 1'b0; s2_valid_ps = 1'b0;
            comp_done_ps = 1'b0;
            pf_slot_ps   = 1'b0; pf_phase_ps = 1'b0;
            pf_done_ps   = (pool_out_dim == 1);      // no next row to prefetch iff single pooled row
            next_state   = CALC;
          end
        end
        // advance the request pipeline: the slot presented this cycle is next cycle's in-flight slot
        if (ld_fetch < NROWS) begin
          ld_inflight_ps = ld_fetch;
          ld_req_ps      = 1'b1;
          ld_fetch_ps    = ld_fetch + 1;
        end else begin
          ld_req_ps      = 1'b0;                      // all requests launched -> drain
        end
`else
        // Clean per-row read pulse (Stage-B fix): request+capture, then a settle cycle.
        if (rd_phase == 0) begin
          mem_intf_read.mem_req        = 1;
          mem_intf_read.mem_start_addr = cur_in_row_addr;
          mem_intf_read.mem_size_bytes = conv_arr_in_dim;
          if (mem_intf_read.mem_valid) begin
            integer i;
            for (i=0;i<DIM_MAX;i++)
              in_buf_ps[load_row][i] = (i<conv_arr_in_dim) ? mem_intf_read.mem_data[i] : 8'd0;
            mem_intf_read.mem_req = 0;
            rd_phase_ps = 1;                         // settle (req low) before the next row
          end
        end else begin
          mem_intf_read.mem_req = 0;
          rd_phase_ps = 0;
          if (load_row == NROWS-1) begin             // slots 0..5 loaded -> start streaming
            // start this pooled row's compute pipeline + prefetch of pr+1's rows
            issue_col_ps = 0;
            s0_valid_ps  = 1'b0; s1_valid_ps = 1'b0; s2_valid_ps = 1'b0;
            comp_done_ps = 1'b0;
            pf_slot_ps   = 1'b0; pf_phase_ps = 1'b0;
            pf_done_ps   = (pool_out_dim == 1);      // no next row to prefetch iff single pooled row
            next_state   = CALC;
          end else begin
            load_row_ps = load_row + 1;              // next initial-load slot
          end
        end
`endif
      end

      CALC: begin
        //=== COMPUTE PIPELINE (identical to the verified Stage C step 3) ===
        // ---- S3 MAXPOOL+STORE : the cell now in 'act' (named by s2) -> one pooled byte ----
        if (s2_valid) begin
          out_row_buf_ps[s2_col] = max4_pool(act[0], act[1], act[2], act[3]);
          if (s2_col == pool_out_dim-1) comp_done_ps = 1'b1;     // last column of the row stored
        end

        // ---- S2 COMBINE+ACTIVATE : from registered gpart -> act_ps ----
        act_ps[0]   = combine_activate(gpart[0], bias_reg);
        act_ps[1]   = combine_activate(gpart[1], bias_reg);
        act_ps[2]   = combine_activate(gpart[2], bias_reg);
        act_ps[3]   = combine_activate(gpart[3], bias_reg);
        s2_col_ps   = s1_col;
        s2_valid_ps = s1_valid;

        // ---- S1 MULT+ROWSUM : from registered win_buf + kernel_buf -> gpart_ps (4 windows) ----
        gpart_ps[0] = mac_rowsums(win_buf[0], kernel_buf);
        gpart_ps[1] = mac_rowsums(win_buf[1], kernel_buf);
        gpart_ps[2] = mac_rowsums(win_buf[2], kernel_buf);
        gpart_ps[3] = mac_rowsums(win_buf[3], kernel_buf);
        s1_col_ps   = s0_col;
        s1_valid_ps = s0_valid;

        // ---- S0 SELECT : mux the four windows of issue_col out of the line buffer -> win_buf_ps ----
        if (issue_col < pool_out_dim) begin
          win_buf_ps[0] = sel_win(in_buf, issue_col, 2'b00);   // (wr,wc)=(0,0)
          win_buf_ps[1] = sel_win(in_buf, issue_col, 2'b01);   // (0,1)
          win_buf_ps[2] = sel_win(in_buf, issue_col, 2'b10);   // (1,0)
          win_buf_ps[3] = sel_win(in_buf, issue_col, 2'b11);   // (1,1)
          s0_col_ps    = issue_col;
          s0_valid_ps  = 1'b1;
          issue_col_ps = issue_col + 1;                        // next pooled column
        end else begin
          s0_valid_ps  = 1'b0;                                 // draining: nothing new selected
        end

        //=== PREFETCH (concurrent): read pr+1's 2 new rows (2pr+6, 2pr+7) into slots 6,7 ===
        // Uses the read port, which the compute pipeline never touches. pf_done already 1 on the
        // last pooled row (no next row), so the port stays idle then.
        if (!pf_done) begin
          if (pf_phase == 0) begin
            mem_intf_read.mem_req        = 1;
            mem_intf_read.mem_start_addr = pf_row_addr;
            mem_intf_read.mem_size_bytes = conv_arr_in_dim;
            if (mem_intf_read.mem_valid) begin
              integer i;
              for (i=0;i<DIM_MAX;i++)
                in_buf_ps[NROWS + pf_slot][i] = (i<conv_arr_in_dim) ? mem_intf_read.mem_data[i] : 8'd0;
              mem_intf_read.mem_req = 0;
              pf_phase_ps = 1'b1;                              // settle before the second prefetch row
            end
          end else begin
            mem_intf_read.mem_req = 0;
            pf_phase_ps = 1'b0;
            if (pf_slot == 1'b1) pf_done_ps = 1'b1;            // both rows (slots 6,7) captured
            else                 pf_slot_ps = 1'b1;            // go capture the second prefetch row
          end
        end

`ifdef CONV_WRITE_OVERLAP
        //=== item 4 WRITE-ENGINE : flush the pending row (out_wbuf) DURING this CALC. Gated on pf_done so
        //    it never co-issues with a prefetch read -> pure write-during-compute. ===
        if (pw_valid && pf_done) begin
          mem_intf_write.mem_req        = 1;
          mem_intf_write.mem_start_addr = pw_out_addr;
          mem_intf_write.mem_size_bytes = pool_out_dim;
          mem_intf_write.mem_data       = out_wbuf;
          if (mem_intf_write.mem_ack) begin
            mem_intf_write.mem_req = 0;
            pw_valid_ps = 1'b0;                              // pending row written -> snapshot buffer free
          end
        end

        //=== item 4 FINISH-ROW : this row computed + prefetched + the previous write drained (!pw_valid).
        //    Snapshot it into out_wbuf, then overlap its write with the NEXT row's CALC. ===
        if (comp_done && pf_done && !pw_valid) begin
          out_wbuf_ps = out_row_buf;                         // snapshot the finished pooled row
          pw_valid_ps = 1'b1;
          pw_pr_ps    = pr;
          if (pr == pool_out_dim-1) begin
            next_state = FINAL_WRITE;                        // last row: no next CALC to overlap -> flush it
          end else begin
            // roll the line buffer up by 2 (reuse 2..5 -> 0..3, prefetched 6,7 -> 4,5) + reinit, stay in CALC
            in_buf_ps[0] = in_buf[2];
            in_buf_ps[1] = in_buf[3];
            in_buf_ps[2] = in_buf[4];
            in_buf_ps[3] = in_buf[5];
            in_buf_ps[4] = in_buf[6];
            in_buf_ps[5] = in_buf[7];
            pr_ps        = pr + 1;
            issue_col_ps = 0;
            s0_valid_ps  = 1'b0; s1_valid_ps = 1'b0; s2_valid_ps = 1'b0;
            comp_done_ps = 1'b0;
            pf_slot_ps   = 1'b0; pf_phase_ps = 1'b0;
            pf_done_ps   = ((pr + 1) == pool_out_dim-1);     // no prefetch on the last pooled row
          end
        end
`else
        //=== EXIT : move to WRITE only when compute AND prefetch are both done ===
        if (s2_valid && (s2_col == pool_out_dim-1) && pf_done) next_state = WRITE; // common: prefetch already done
        else if (comp_done && pf_done)                         next_state = WRITE; // compute finished earlier, prefetch caught up
`endif
      end

      WRITE: begin
        mem_intf_write.mem_req = 1;
        if (mem_intf_write.mem_ack) begin
          mem_intf_write.mem_req = 0;
          if (pr == pool_out_dim-1) next_state = DONE;             // last pooled row -> done
          else begin
            // Roll the line buffer up by 2: reuse rows 2..5 -> 0..3, and the PREFETCHED rows 6,7 -> 4,5.
            in_buf_ps[0] = in_buf[2];
            in_buf_ps[1] = in_buf[3];
            in_buf_ps[2] = in_buf[4];
            in_buf_ps[3] = in_buf[5];
            in_buf_ps[4] = in_buf[6];    // prefetched new row (2*pr+6)
            in_buf_ps[5] = in_buf[7];    // prefetched new row (2*pr+7)
            pr_ps       = pr + 1;
            // Re-init the compute pipeline + prefetch for the NEXT pooled row -> go straight to CALC.
            issue_col_ps = 0;
            s0_valid_ps  = 1'b0; s1_valid_ps = 1'b0; s2_valid_ps = 1'b0;
            comp_done_ps = 1'b0;
            pf_slot_ps   = 1'b0; pf_phase_ps = 1'b0;
            pf_done_ps   = ((pr + 1) == pool_out_dim-1);   // no prefetch on the last pooled row
            next_state   = CALC;
          end
        end
      end

`ifdef CONV_WRITE_OVERLAP
      // item 4: flush the LAST pooled row (no next CALC to overlap with), then finish.
      FINAL_WRITE: begin
        if (pw_valid) begin
          mem_intf_write.mem_req        = 1;
          mem_intf_write.mem_start_addr = pw_out_addr;
          mem_intf_write.mem_size_bytes = pool_out_dim;
          mem_intf_write.mem_data       = out_wbuf;
          if (mem_intf_write.mem_ack) begin
            mem_intf_write.mem_req = 0;
            pw_valid_ps = 1'b0;
            next_state  = DONE;
          end
        end
      end
`endif

      DONE: begin
        conv_done = 1;
        if (clear_done_on_read) next_state = IDLE;
      end

    endcase

  end // always_comb

  //--------------------------------------------------------------------------------------------------------
  // Sequential

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state       <= IDLE;
      pr          <= 0;
      load_row    <= 0;
      rd_phase    <= 0;
      ld_fetch    <= 0;
      ld_inflight <= 0;
      ld_req      <= 0;
      issue_col   <= 0;
      s0_col      <= 0; s0_valid <= 0;
      s1_col      <= 0; s1_valid <= 0;
      s2_col      <= 0; s2_valid <= 0;
      comp_done   <= 0;
      pf_phase    <= 0;
      pf_slot     <= 0;
      pf_done     <= 0;
      kernel_buf  <= 0;
      in_buf      <= 0;
      out_row_buf <= 0;
      out_wbuf    <= 0;
      pw_valid    <= 0;
      pw_pr       <= 0;
      bias_reg    <= 0;
      win_buf     <= '0;
      gpart       <= '0;
      act         <= '0;
    end else begin
      state       <= next_state;
      pr          <= pr_ps;
      load_row    <= load_row_ps;
      rd_phase    <= rd_phase_ps;
      ld_fetch    <= ld_fetch_ps;
      ld_inflight <= ld_inflight_ps;
      ld_req      <= ld_req_ps;
      issue_col   <= issue_col_ps;
      s0_col      <= s0_col_ps; s0_valid <= s0_valid_ps;
      s1_col      <= s1_col_ps; s1_valid <= s1_valid_ps;
      s2_col      <= s2_col_ps; s2_valid <= s2_valid_ps;
      comp_done   <= comp_done_ps;
      pf_phase    <= pf_phase_ps;
      pf_slot     <= pf_slot_ps;
      pf_done     <= pf_done_ps;
      kernel_buf  <= kernel_buf_ps;
      in_buf      <= in_buf_ps;
      out_row_buf <= out_row_buf_ps;
      out_wbuf    <= out_wbuf_ps;
      pw_valid    <= pw_valid_ps;
      pw_pr       <= pw_pr_ps;
      bias_reg    <= bias_reg_ps;
      win_buf     <= win_buf_ps;
      gpart       <= gpart_ps;
      act         <= act_ps;
    end
  end

  //--------------------------------------------------------------------------------------------------------
  // Comb function (S0): extract one 5x5 conv window's 25 input bytes from the line buffer.
  // win_in[1]=wr (top/bottom conv row), win_in[0]=wc (left/right conv col). The window's top-left
  // conv position is (2*pr+wr, 2*pc+wc); in the line buffer that is in_buf[wr+r][2*pc+wc+c].

  function automatic logic [KSIZE-1:0][7:0] sel_win;
    input logic [BUF_ROWS-1:0][DIM_MAX-1:0][7:0] inb;
    input logic [ARR_IDX_W:0]                    pc_in;
    input logic [1:0]                            win_in;   // [1]=wr , [0]=wc
    integer r, c, wr_base, col_base;
    begin
      wr_base  = win_in[1];               // 0 = top conv row, 1 = bottom conv row
      col_base = (pc_in<<1) + win_in[0];  // 2*pc + wc = left conv column of this window
      for (r=0;r<KDIM;r++)
        for (c=0;c<KDIM;c++)
          sel_win[r*KDIM+c] = inb[wr_base+r][col_base+c];
    end
  endfunction

  //--------------------------------------------------------------------------------------------------------
  // Comb function (S1): 5 row partial sums of one 5x5 conv window from its registered 25 input bytes.
  // Same signed products and grouping as the golden conv_window_val_nox (associative -> bit-exact).

  function automatic logic [KDIM-1:0][ACC_W-1:0] mac_rowsums;
    input logic [KSIZE-1:0][7:0] win_bytes;    // this window's 25 input bytes (registered)
    input logic [KSIZE-1:0][7:0] kern;         // 25 int8 kernel weights
    integer r, c;
    logic signed [8:0]       in_s, wgt_s;
    logic signed [ACC_W-1:0] rowsum;
    begin
      for (r=0;r<KDIM;r++) begin
        rowsum = 0;
        for (c=0;c<KDIM;c++) begin
          in_s   = $signed({1'b0, win_bytes[r*KDIM+c]});         // uint8  -> +9b (zero-extend)
          wgt_s  = $signed({kern[r*KDIM+c][7], kern[r*KDIM+c]}); // int8 sign-extended -> 9b
          rowsum = rowsum + (in_s * wgt_s);
        end
        mac_rowsums[r] = rowsum;
      end
    end
  endfunction

  //--------------------------------------------------------------------------------------------------------
  // Comb function (S2): combine the five row partial sums + bias, then activate (one window).

  function automatic logic [7:0] combine_activate;
    input logic [KDIM-1:0][ACC_W-1:0] g;
    input logic signed [ACC_W-1:0]    bias_in;
    integer r;
    logic signed [ACC_W-1:0] acc;
    begin
      acc = bias_in;
      for (r=0;r<KDIM;r++) acc = acc + $signed(g[r]);
      combine_activate = relu_descale(acc);
    end
  endfunction

  //--------------------------------------------------------------------------------------------------------
  // ReLU + descale(/256) + saturate(255). Bit-identical to relu_and_descale() in the C model.

  function automatic logic [7:0] relu_descale;
    input logic signed [ACC_W-1:0] acc_in;
    logic signed [ACC_W-1:0] a, d;
    begin
      a = acc_in;
      if (a < 0) a = 0;                 // ReLU (acc <= 0 -> 0)
      d = a >>> 8;                      // descale by 256
      if (d > 255) relu_descale = 8'd255;
      else         relu_descale = d[7:0];
    end
  endfunction

  //--------------------------------------------------------------------------------------------------------
  // 2x2 max-pool of four activations (S3) -- same reduction as pool.sv max4_pool.
  // Window order: v0=win0=(0,0) v1=win1=(0,1) v2=win2=(1,0) v3=win3=(1,1).

  function automatic logic [7:0] max4_pool;
    input logic [7:0] v0, v1, v2, v3;
    logic [7:0] m01, m23;
    begin
      m01 = v0 > v1 ? v0 : v1;
      m23 = v2 > v3 ? v2 : v3;
      max4_pool = m01 > m23 ? m01 : m23;
    end
  endfunction

endmodule
