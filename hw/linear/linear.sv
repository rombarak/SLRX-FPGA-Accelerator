import xbox_def_pkg::*;
import slrx_def_pkg::*;

//--------------------------------------------------------------------------------------------------------
// Linear (fully-connected) accelerator, HARDWARE-LOOPED, with a PIPELINED load/compute/store path and a
// FUSED hardware argmax ("Select").
//
//  LIN_SETUP : reads the input vector ONCE (stationary) and BULK-reads the WHOLE bias vector (int32[N])
//              in a few 32-byte line reads into bias_buf -> CALC has zero per-element bias traffic.
//  LIN_CALC  : streams one weight row per output element (read row -> register -> 1-cycle combinational MAC
//              from the registered row, so the MAC critical path stays register->register), accumulates the
//              whole output vector on-chip, then writes it back as ONE block (like pool.sv). When sel_en is
//              set (Linear1 only), a running argmax is folded into the stream (0 extra cycles, short path)
//              and the winning index is written as one byte to a CPU-supplied XMEM scratch (no SW Select).
//
// Register reuse (no new command / register / platform-interface change):
//   OUT_COL_IDX_RI (10) -> sel_en   (bit0): 1 for Linear1, 0 for Linear0 (both unused by linear before).
//   OUT_ROW_IDX_RI (9)  -> result_addr: XMEM byte the accelerator writes the argmax index to.
//
// The dot product (calc_lin_element) and relu_and_descale arithmetic are UNCHANGED (bit-exact vs the C
// golden lin_elem_nox); only the bias schedule (bulk vs per-element), the output packing (block vs byte)
// and the fused argmax are new. The argmax replicates get_max_val_idx EXACTLY, including its int8_t
// signedness quirk (SELECT_SIGNED_QUIRK), so the detected label is unchanged.
// Known-good pre-Select version kept at linear.sv.preselect.bak.
//--------------------------------------------------------------------------------------------------------

// ============================================================================================
//  LIN_BACKTOBACK — (V) CLOUD-VERIFIED 2026-06-28: Total 1,027, slrx 51.29 MHz, ITT ~20.0 us,
//  LIN1 0/27, "we are the champions" 20/20. Held-high mem_req + a NEW address every cycle WORKS:
//  the muxed read port is a LATENCY-1 PIPELINE -- the data on a mem_valid cycle is for the address
//  requested ONE cycle EARLIER, so arriving rows are bound to the LAGGING inflight_idx (NOT the
//  address on the bus now). The earlier "PROVEN DEAD / 21/27 mismatches" was that OFF-BY-ONE, NOT a
//  dead port; the settle is NOT mandatory. STREAM ~3 -> ~1 cyc/elem. Leaving this `define ON is the
//  verified keeper. (Same trick revives conv item 5 / F2 -- back-to-back conv row reads.)
`define LIN_BACKTOBACK
// ============================================================================================

// ============================================================================================
//  LIN_MACPIPE (Stage 3, Fmax) — pipeline the flat 32-MAC into registered stages so Quartus packs the
//  32 multiplies into DSP I/O registers (conv-style): S2 = 4 partials of 8 taps -> S3 = bias-add +
//  ReLU/>>>8/sat -> S4 = store + argmax-feed, with a 2-deep {valid,idx} tag. Bit-exact (re-association
//  at ACC_W=32, no intermediate truncation). VERIFY on -fit that ~32 mults land in DSP, not LE.
//  Comment out to fall back to the single-cycle MAC keeper (back-to-back F1): combined best Total 927 @
//  slrx 51.78 MHz, ITT 17.9µs WITH the prefetch conv (1,027 @ 51.29 when paired with the 4-MAC conv).
`define LIN_MACPIPE
// ============================================================================================

module linear (
  input   clk,
  input   rst_n,

  slrx_regs_intrf.xlr slrx_regs_intrf, // Host Registers Interface

  // muxed interfaces
  mem_intf_read.client_read   mem_intf_read,
  mem_intf_write.client_write mem_intf_write
);

  enum {IDLE, SETUP_IN, SETUP_BIAS, STREAM, WRITE, SELECT, WRITE_IDX, DONE} next_state, state;

  localparam DIM_MAX_SIZE      = 32;                       // all dims <= 32
  localparam MAX_DOT_PROD_WIDTH = 16 + $clog2(DIM_MAX_SIZE);// 21 : multiplied byte width (8+8) + #elements
  localparam ARR_IDX_W         = $clog2(DIM_MAX_SIZE);     // 5
  localparam SELECT_SIGNED_QUIRK = 1;                      // 1: match get_max_val_idx's int8_t max_val exactly
  localparam ACC_W = 32;                                  // 32-bit signed partials/acc (matches C int32 golden; conv ACC_W)

  // ---- host-reg decoded inputs ----
  slrx_cmd_t slrx_cmd;
  logic lin_active, lin_start, lin_done, clear_done_on_read;

  logic [XMEM_ADDR_WIDTH-1:0] lin_wgt_arr_addr;
  logic [XMEM_ADDR_WIDTH-1:0] lin_arr_in_addr;
  logic [XMEM_ADDR_WIDTH-1:0] lin_arr_out_addr;
  logic [XMEM_ADDR_WIDTH-1:0] lin_bias_vec_addr;
  logic [XMEM_ADDR_WIDTH-1:0] sel_result_addr;             // OUT_ROW_IDX_RI : where to write the argmax index
  logic                       sel_en;                      // OUT_COL_IDX_RI[0] : enable fused Select (Linear1)
  logic                       fused_setup;                 // OUT_COL_IDX_RI[1] : LIN_CALC self-runs SETUP (skip the separate LIN_SETUP command)
  logic                       fused_flow, fused_flow_ps;   // 1 while a fused LIN_CALC is doing its internal SETUP -> fall through to STREAM

  logic [ARR_IDX_W:0] lin_arr_in_dim;                      // input vector length
  logic [ARR_IDX_W:0] lin_arr_out_dim;                     // output vector length

  // ---- on-chip buffers (stationary across LIN_SETUP -> LIN_CALC) ----
  logic [DIM_MAX_SIZE-1:0][7:0]  in_vec,  in_vec_ps;       // input vector (read once)
  logic [DIM_MAX_SIZE-1:0][31:0] bias_buf, bias_buf_ps;    // whole bias vector (int32), bulk-read
  logic [DIM_MAX_SIZE-1:0][7:0]  wgt_vec, wgt_vec_ps;      // current weight row (registered before the MAC)
  logic [DIM_MAX_SIZE-1:0][7:0]  out_buf, out_buf_ps;      // accumulated output vector (block-written)

  // ---- streaming / control counters ----
  logic [ARR_IDX_W-1:0] rd_idx,      rd_idx_ps;            // output element / weight row being processed
  logic [2:0]           bias_rd_idx, bias_rd_idx_ps;       // bias 32-byte chunk index (0..3)
  logic                 rd_phase,    rd_phase_ps;          // read sub-cycle (0: request+capture, 1: settle/compute)

  // ---- fused argmax (running) ----
  logic [ARR_IDX_W-1:0] best_idx, best_idx_ps;             // index of the running max
  logic [7:0]           best_byte, best_byte_ps;           // value at the running-max index

  // ---- argmax ONE STAGE BEHIND the MAC : compare a REGISTERED element value, parallel to the MAC ----
  logic [7:0]           last_val,  last_val_ps;            // mac_out of the element just registered (out_buf[i])
  logic [ARR_IDX_W-1:0] last_idx,  last_idx_ps;            // its index i
  logic                 have_last, have_last_ps;           // a captured-but-not-yet-compared element exists

  // ---- overlapped fetch/compute pipeline (default STREAM): compute the registered row while the next
  //      row's read is OUTSTANDING. ONE read in flight (mem_req held until mem_valid, cgol-style), so the
  //      one-at-a-time XMEM protocol is respected -- this is NOT the dead back-to-back (held-high) probe.
  logic [ARR_IDX_W:0]   fetch_idx, fetch_idx_ps;           // next weight row to REQUEST (0..N, 6-bit)
  logic [ARR_IDX_W-1:0] comp_idx,  comp_idx_ps;            // index of the row registered in wgt_vec (compute target)
  logic                 have_row,  have_row_ps;            // wgt_vec/cur_bias hold a row to compute this cycle

  // ---- HELD-HIGH read pipeline : the data on a mem_valid cycle belongs to the address requested ONE
  //      cycle EARLIER (read latency 1), NOT the address on the bus now. inflight_idx tracks that lagging
  //      index so the arriving row is bound correctly (the off-by-one that gave 21/27 mismatches before).
  logic [ARR_IDX_W:0]   inflight_idx, inflight_idx_ps;     // index whose data arrives on the NEXT mem_valid
  logic                 req_inflight, req_inflight_ps;     // a request has been issued and not yet returned

  // ---- LIN_MACPIPE 32-MAC pipeline : S2 partials -> S3 activation -> S4 store, 2-deep {valid,idx} tag ----
  logic [3:0][ACC_W-1:0] psum,    psum_ps;                 // S2: 4 partials of 8 taps (registered DSP-output stage)
  logic                  p_valid, p_valid_ps;              // S2 tag valid
  logic [ARR_IDX_W-1:0]  p_idx,   p_idx_ps;                // S2 tag idx (aligned with psum; bias read by this in S3)
  logic [7:0]            act,     act_ps;                  // S3 activated result (ReLU/>>>8/sat)
  logic                  a_valid, a_valid_ps;              // S3 tag valid
  logic [ARR_IDX_W-1:0]  a_idx,   a_idx_ps;                // S3 tag idx (aligned with act; out_buf written by this in S4)

  // ---- registered current-element bias (pulls the 32-way bias mux OFF the MAC critical path -> Fmax) ----
  logic signed [MAX_DOT_PROD_WIDTH-1:0] cur_bias, cur_bias_ps;
  logic [ARR_IDX_W-1:0] sel_idx, sel_idx_ps;               // de-folded argmax scan index (over out_buf)

  // ---- combinational helpers ----
  logic [7:0]          mac_out;                            // this element's activated result
  logic signed [8:0]   cand9, best9;                       // 9-bit signed compare operands for the argmax
  logic [ARR_IDX_W:0]  bias_ints_left;                     // int32 still to read at this chunk
  logic [ARR_IDX_W:0]  bias_ints_this;                     // int32 in this chunk (<=8)
  logic [6:0]          bias_size_this;                     // bytes in this chunk (<=32)

  //--------------------------------------------------------------------------------------------------------
  // Host Regs Interface

  assign slrx_regs_intrf.xlr_done = lin_done;

  assign slrx_cmd           = slrx_cmd_t'(slrx_regs_intrf.host_regs[XLR_START_RI][$clog2(NUM_SLRX_CMDS)-1:0]);
  assign lin_active         = (slrx_cmd==LIN_SETUP) || (slrx_cmd==LIN_CALC);
  assign lin_start          = slrx_regs_intrf.host_regs_valid_pulse[XLR_START_RI] && lin_active;
  assign clear_done_on_read = lin_active && slrx_regs_intrf.xlr_done_ack;

  assign lin_wgt_arr_addr   = slrx_regs_intrf.host_regs[WGT_ADDR_RI];
  assign lin_bias_vec_addr  = slrx_regs_intrf.host_regs[LIN_BIAS_ADDR_RI];
  assign lin_arr_in_addr    = slrx_regs_intrf.host_regs[ARR_IN_ADDR_RI];
  assign lin_arr_out_addr   = slrx_regs_intrf.host_regs[ARR_OUT_ADDR_RI];
  assign lin_arr_in_dim     = slrx_regs_intrf.host_regs[ARR_IN_DIM_RI];
  assign lin_arr_out_dim    = slrx_regs_intrf.host_regs[ARR_OUT_DIM_RI];
  assign sel_result_addr    = slrx_regs_intrf.host_regs[OUT_ROW_IDX_RI];
  assign sel_en             = slrx_regs_intrf.host_regs[OUT_COL_IDX_RI][0];
  assign fused_setup        = slrx_regs_intrf.host_regs[OUT_COL_IDX_RI][1];

  //========================================================================================================
  // State machine + datapath (combinational)
  always_comb begin

    // ---- defaults ----
    next_state = state;

    rd_idx_ps      = rd_idx;
    bias_rd_idx_ps = bias_rd_idx;
    rd_phase_ps    = rd_phase;
    best_idx_ps    = best_idx;
    best_byte_ps   = best_byte;
    cur_bias_ps    = cur_bias;
    sel_idx_ps     = sel_idx;
    last_val_ps    = last_val;
    last_idx_ps    = last_idx;
    have_last_ps   = have_last;
    fused_flow_ps  = fused_flow;
    fetch_idx_ps   = fetch_idx;
    comp_idx_ps    = comp_idx;
    have_row_ps    = have_row;
    inflight_idx_ps = inflight_idx;
    req_inflight_ps = req_inflight;
    psum_ps     = psum;
    p_valid_ps  = 1'b0;                                    // pipeline valids default LOW: driven each STREAM cycle, empty elsewhere
    p_idx_ps    = p_idx;
    act_ps      = act;
    a_valid_ps  = 1'b0;
    a_idx_ps    = a_idx;

    in_vec_ps   = in_vec;
    bias_buf_ps = bias_buf;
    wgt_vec_ps  = wgt_vec;
    out_buf_ps  = out_buf;

    mem_intf_read.mem_req        = 0;
    mem_intf_read.mem_start_addr = 0;
    mem_intf_read.mem_size_bytes = 0;

    mem_intf_write.mem_req        = 0;
    mem_intf_write.mem_start_addr = 0;
    mem_intf_write.mem_size_bytes = 0;
    mem_intf_write.mem_data       = 0;

    lin_done = 0;

    mac_out = 0;
    cand9   = 0;
    best9   = 0;

    // bias chunk sizing : up to 8 int32 (32 bytes) per read, clamp the last chunk (no over-read).
    // NB: use '*' not '<<' on bias_rd_idx (a shift's result width = the 3-bit operand width -> would truncate).
    bias_ints_left = lin_arr_out_dim - (bias_rd_idx * 8);
    bias_ints_this = (bias_ints_left > 8) ? 8 : bias_ints_left;
    bias_size_this = bias_ints_this * 4;

    case (state)

      IDLE: begin
        if (lin_start) begin
          if (slrx_cmd==LIN_SETUP) begin
            rd_phase_ps   = 0;
            fused_flow_ps = 0;                              // legacy separate-SETUP command
            next_state    = SETUP_IN;
          end else if (slrx_cmd==LIN_CALC) begin
            rd_idx_ps    = 0;
            rd_phase_ps  = 0;
            best_idx_ps  = 0;
            best_byte_ps = 0;
            have_last_ps = 0;                               // no element captured for the behind-MAC argmax yet
            fetch_idx_ps = 0;
            comp_idx_ps  = 0;
            have_row_ps  = 0;
            inflight_idx_ps = 0;
            req_inflight_ps = 0;
            // Stage 2B command fusion: if OUT_COL_IDX_RI[1] is set, this single LIN_CALC also runs SETUP
            // (read input vec + bulk bias) first -> the CPU issues ONE command + ONE poll per layer.
            if (fused_setup) begin
              fused_flow_ps = 1;                            // SETUP_BIAS will fall through to STREAM
              next_state    = SETUP_IN;                     // read input vec + bulk bias inside this LIN_CALC
            end else begin
              fused_flow_ps = 0;                            // legacy: SETUP already ran via LIN_SETUP
              next_state    = STREAM;
            end
          end
        end
      end

      // ---- LIN_SETUP : read input vector once, then bulk-read the whole bias vector ----
      SETUP_IN: begin
        mem_intf_read.mem_req        = 1;
        mem_intf_read.mem_start_addr = lin_arr_in_addr;
        mem_intf_read.mem_size_bytes = lin_arr_in_dim;
        if (mem_intf_read.mem_valid) begin
          integer i;
          for (i=0;i<DIM_MAX_SIZE;i++)
            in_vec_ps[i] = (i<lin_arr_in_dim) ? mem_intf_read.mem_data[i] : 8'd0;
          mem_intf_read.mem_req = 0;
          bias_rd_idx_ps = 0;
          rd_phase_ps    = 0;
          next_state     = SETUP_BIAS;
        end
      end

      SETUP_BIAS: begin
        if (rd_phase == 0) begin
          mem_intf_read.mem_req        = 1;
          mem_intf_read.mem_start_addr = lin_bias_vec_addr + (bias_rd_idx * 32); // 32 bytes / chunk
          mem_intf_read.mem_size_bytes = bias_size_this;
          if (mem_intf_read.mem_valid) begin
            integer j;
            for (j=0;j<8;j++)
              if ((bias_rd_idx * 8) + j < lin_arr_out_dim)
                bias_buf_ps[(bias_rd_idx * 8) + j] = { mem_intf_read.mem_data[4*j+3],
                                                       mem_intf_read.mem_data[4*j+2],
                                                       mem_intf_read.mem_data[4*j+1],
                                                       mem_intf_read.mem_data[4*j+0] };
            mem_intf_read.mem_req = 0;
            rd_phase_ps = 1;                                  // settle (req low) before the next chunk
          end
        end else begin
          mem_intf_read.mem_req = 0;
          rd_phase_ps = 0;
          if (((bias_rd_idx + 1) * 8) >= lin_arr_out_dim)
            next_state = fused_flow ? STREAM : DONE;        // fused: fall through to the MAC stream; legacy: done
          else bias_rd_idx_ps = bias_rd_idx + 1;
        end
      end

      // ---- LIN_CALC : stream weight rows, MAC from the registered row, running argmax, accumulate ----
      STREAM: begin
`ifdef LIN_BACKTOBACK
        // ===== Stage A probe: BACK-TO-BACK weight reads (NO settle). Fetch row[fetch_idx] every cycle;
        //   compute the row registered last cycle (wgt_vec) so the MAC stays register->register; argmax
        //   runs one more stage behind. Capture is GATED on mem_valid -> if the port needs a settle the
        //   OUTPUT stays correct (just no speed-up); if held-high mem_req streams new rows it's ~1
        //   cyc/element; if it deadlocks it hangs -> rebuild with the macro OFF. =====
        // argmax on the element computed last cycle (registered last_val), off the MAC cone
        if (have_last) begin
          cand9 = $signed({1'b0, last_val});
          best9 = SELECT_SIGNED_QUIRK ? $signed({best_byte[7], best_byte})
                                      : $signed({1'b0,          best_byte});
          if (last_idx == 0)      begin best_idx_ps = 0;        best_byte_ps = last_val; end
          else if (cand9 > best9) begin best_idx_ps = last_idx; best_byte_ps = last_val; end
          have_last_ps = 1'b0;
        end
        // COMPUTE the row captured last cycle. have_row CLEARS each cycle (set again only by a fresh
        // capture below) -- so a cycle with no new valid computes nothing and have_row falls to 0.
`ifdef LIN_MACPIPE
        // ===== 32-MAC PIPELINE (Stage 3, Fmax). S2 mult+group / S3 combine+activate / S4 store, with a
        //   2-deep {valid,idx} tag. issue = have_row (a fresh row this cycle). The pipe drains ~2 cycles
        //   after the last issue; WRITE fires when the LAST element (a_idx==N-1) exits S4. =====
        // S2 : 4 partials of 8 taps from the registered weight row (registered operands -> DSP). Tag t1.
        if (have_row) begin
          psum_ps     = mac_group_partials(wgt_vec, in_vec);
          p_valid_ps  = 1'b1;
          p_idx_ps    = comp_idx;
          have_row_ps = 1'b0;                               // consumed; re-armed only by a fresh capture
        end
        // S3 : acc = bias_buf[p_idx] + sum(psum) (32-bit signed), then ReLU/>>>8/sat. Bias indexed HERE
        //      (Correction 2) -> off the MAC cone; no pipelined cur_bias -> no bias/data skew.
        begin
          logic signed [ACC_W-1:0] acc3, descaled3;
          acc3 = $signed(bias_buf[p_idx][MAX_DOT_PROD_WIDTH-1:0])
               + $signed(psum[0]) + $signed(psum[1]) + $signed(psum[2]) + $signed(psum[3]);
          if (acc3 < 0) acc3 = 0;                           // ReLU (verbatim)
          descaled3 = acc3 >>> 8;                           // descale /256 (verbatim)
          act_ps     = (descaled3 > 255) ? 8'd255 : descaled3[7:0]; // saturate (verbatim)
          a_valid_ps = p_valid;                             // advance the tag
          a_idx_ps   = p_idx;
        end
        // S4 : store the registered activation, feed the behind-MAC argmax, terminate on the last element.
        if (a_valid) begin
          out_buf_ps[a_idx] = act;
          last_val_ps = act;  last_idx_ps = a_idx;  have_last_ps = 1'b1;
          if (a_idx == lin_arr_out_dim - 1) next_state = WRITE; // last element exited S4 -> stream done
        end
`else
        // single-cycle MAC (the verified 1,027 keeper). LIN_MACPIPE off = this path.
        if (have_row) begin
          mac_out = calc_lin_element(wgt_vec, cur_bias, in_vec);
          out_buf_ps[comp_idx] = mac_out;
          last_val_ps = mac_out;  last_idx_ps = comp_idx;  have_last_ps = 1'b1;
          have_row_ps = 1'b0;                               // consumed; re-set only by a new capture
          if (comp_idx == lin_arr_out_dim - 1) next_state = WRITE; // last element computed -> stream done
        end
`endif

        // FETCH (held-high): present a NEW address every cycle while rows remain to request. The DATA that
        // returns on a mem_valid cycle belongs to the address requested ONE cycle earlier (latency 1), i.e.
        // to inflight_idx -- NOT to fetch_idx, which is already the NEXT address on the bus. Binding to
        // inflight_idx removes the off-by-one that mislabeled every row (the 21/27-mismatch failure).
        if (fetch_idx < lin_arr_out_dim) begin
          mem_intf_read.mem_req        = 1;
          mem_intf_read.mem_start_addr = lin_wgt_arr_addr + (fetch_idx * lin_arr_in_dim);
          mem_intf_read.mem_size_bytes = lin_arr_in_dim;
        end

        // Capture whatever returns this cycle and bind it to the IN-FLIGHT index (the lagging one).
        // req_inflight gates it so a valid before any request has truly launched is ignored.
        if (mem_intf_read.mem_valid && req_inflight) begin
          integer i;
          for (i=0;i<DIM_MAX_SIZE;i++)
            wgt_vec_ps[i] = (i<lin_arr_in_dim) ? mem_intf_read.mem_data[i] : 8'd0;
          cur_bias_ps = bias_buf[inflight_idx[ARR_IDX_W-1:0]][MAX_DOT_PROD_WIDTH-1:0];
          comp_idx_ps = inflight_idx[ARR_IDX_W-1:0];
          have_row_ps = 1'b1;                               // re-arm compute for next cycle
        end

        // Advance the request pipeline: the address presented this cycle is next cycle's in-flight idx.
        if (fetch_idx < lin_arr_out_dim) begin
          inflight_idx_ps = fetch_idx;
          req_inflight_ps = 1'b1;                            // a request is now outstanding
          fetch_idx_ps    = fetch_idx + 1;
        end else begin
          req_inflight_ps = 1'b0;                            // no more requests to launch -> drain
        end
`else
        if (rd_phase == 0) begin
`ifndef LIN_DEFOLDED_SELECT
          // ARGMAX ONE STAGE BEHIND THE MAC: compare the element captured last cycle (last_val, a REGISTER),
          // in parallel with issuing this element's read request. Path = reg -> 9-bit signed compare -> reg,
          // entirely off the wgt_vec->Mult->add-tree MAC cone, so Fmax is unaffected. have_last gates it to
          // exactly once per element (it self-clears, so multi-cycle mem_valid waits don't re-compare).
          if (have_last) begin
            cand9 = $signed({1'b0, last_val});
            best9 = SELECT_SIGNED_QUIRK ? $signed({best_byte[7], best_byte})
                                        : $signed({1'b0,          best_byte});
            if (last_idx == 0)      begin best_idx_ps = 0;        best_byte_ps = last_val; end // init at elem 0
            else if (cand9 > best9) begin best_idx_ps = last_idx; best_byte_ps = last_val; end // strict > : first wins
            have_last_ps = 1'b0;
          end
`endif
          mem_intf_read.mem_req        = 1;
          mem_intf_read.mem_start_addr = lin_wgt_arr_addr + (rd_idx * lin_arr_in_dim);
          mem_intf_read.mem_size_bytes = lin_arr_in_dim;
          if (mem_intf_read.mem_valid) begin
            integer i;
            for (i=0;i<DIM_MAX_SIZE;i++)
              wgt_vec_ps[i] = (i<lin_arr_in_dim) ? mem_intf_read.mem_data[i] : 8'd0;
            cur_bias_ps = bias_buf[rd_idx][MAX_DOT_PROD_WIDTH-1:0]; // register this element's bias (off the MAC path)
            mem_intf_read.mem_req = 0;
            rd_phase_ps = 1;                                  // next cycle: compute from the registered row + bias
          end
        end else begin
          mem_intf_read.mem_req = 0;
          rd_phase_ps = 0;
          // 1-cycle combinational MAC from the REGISTERED weight row + bias (path stays register->register).
          // NB: the cand>best comparator is NOT in this cycle -- folding it onto the combinational mac_out put
          // it IN SERIES with the 32-MAC (STA worst path: wgt_vec -> Mult -> Add-tree -> LessThan45 ->
          // best_byte) and dropped Fmax to ~47. Instead we only CAPTURE mac_out into a register here; the
          // compare runs one stage behind, next cycle, off the MAC cone (see rd_phase==0 / WRITE).
          mac_out = calc_lin_element(wgt_vec, cur_bias, in_vec);
          out_buf_ps[rd_idx] = mac_out;
`ifndef LIN_DEFOLDED_SELECT
          last_val_ps = mac_out;  last_idx_ps = rd_idx;  have_last_ps = 1'b1; // hand this element to the behind-MAC argmax
`endif
          if (rd_idx == lin_arr_out_dim - 1) next_state = WRITE;
          else rd_idx_ps = rd_idx + 1;
        end
`endif
      end

      // ---- write the whole output vector as ONE block ----
      WRITE: begin
        mem_intf_write.mem_req        = 1;
        mem_intf_write.mem_start_addr = lin_arr_out_addr;
        mem_intf_write.mem_size_bytes = lin_arr_out_dim;
        mem_intf_write.mem_data       = out_buf;
`ifndef LIN_DEFOLDED_SELECT
        // TAIL: element N-1 was captured in its rd_phase==1 but has no following rd_phase==0, so compare it
        // here -- on a cycle WRITE already burns waiting for mem_ack (free). have_last self-clears so a
        // multi-cycle mem_ack can't double-apply.
        if (sel_en && have_last) begin
          cand9 = $signed({1'b0, last_val});
          best9 = SELECT_SIGNED_QUIRK ? $signed({best_byte[7], best_byte})
                                      : $signed({1'b0,          best_byte});
          if (last_idx == 0)      begin best_idx_ps = 0;        best_byte_ps = last_val; end // N==1 corner
          else if (cand9 > best9) begin best_idx_ps = last_idx; best_byte_ps = last_val; end
          have_last_ps = 1'b0;
        end
`endif
        if (mem_intf_write.mem_ack) begin
          mem_intf_write.mem_req = 0;
`ifdef LIN_DEFOLDED_SELECT
          if (sel_en) begin sel_idx_ps = 0; next_state = SELECT; end // fallback: separate N-cycle argmax scan
          else                              next_state = DONE;
`else
          if (sel_en) next_state = WRITE_IDX;                        // argmax already done behind the MAC
          else        next_state = DONE;
`endif
        end
      end

`ifdef LIN_DEFOLDED_SELECT
      // ---- FALLBACK (define LIN_DEFOLDED_SELECT): de-folded argmax as a separate ~N-cycle scan over the
      //      REGISTERED out_buf (bit-exact to get_max_val_idx, incl. the int8_t quirk). Costs ~N cycles.
      //      The default build instead runs this argmax ONE STAGE BEHIND the MAC (0 extra cycles). ----
      SELECT: begin
        cand9 = $signed({1'b0, out_buf[sel_idx]});
        best9 = SELECT_SIGNED_QUIRK ? $signed({best_byte[7], best_byte})
                                    : $signed({1'b0,          best_byte});
        if (sel_idx == 0) begin
          best_idx_ps  = 0;
          best_byte_ps = out_buf[0];
        end else if (cand9 > best9) begin
          best_idx_ps  = sel_idx;
          best_byte_ps = out_buf[sel_idx];
        end
        if (sel_idx == lin_arr_out_dim - 1) next_state = WRITE_IDX;
        else sel_idx_ps = sel_idx + 1;
      end
`endif

      // ---- write the 1-byte argmax index to the CPU-supplied scratch ----
      WRITE_IDX: begin
        mem_intf_write.mem_req        = 1;
        mem_intf_write.mem_start_addr = sel_result_addr;
        mem_intf_write.mem_size_bytes = 1;
        mem_intf_write.mem_data       = best_idx;            // byte 0 = winning index (0..N-1)
        if (mem_intf_write.mem_ack) begin
          mem_intf_write.mem_req = 0;
          next_state = DONE;
        end
      end

      DONE: begin
        lin_done = 1;
        if (clear_done_on_read) next_state = IDLE;
      end

    endcase

  end // always_comb

  //--------------------------------------------------------------------------------------------------------
  // Sequential

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state       <= IDLE;
      rd_idx      <= 0;
      bias_rd_idx <= 0;
      rd_phase    <= 0;
      best_idx    <= 0;
      best_byte   <= 0;
      cur_bias    <= 0;
      sel_idx     <= 0;
      last_val    <= 0;
      last_idx    <= 0;
      have_last   <= 0;
      fused_flow  <= 0;
      fetch_idx   <= 0;
      comp_idx    <= 0;
      have_row    <= 0;
      inflight_idx <= 0;
      req_inflight <= 0;
      psum    <= '0;
      p_valid <= 0;
      p_idx   <= 0;
      act     <= 0;
      a_valid <= 0;
      a_idx   <= 0;
      in_vec      <= 0;
      bias_buf    <= '0;
      wgt_vec     <= 0;
      out_buf     <= 0;
    end else begin
      state       <= next_state;
      rd_idx      <= rd_idx_ps;
      bias_rd_idx <= bias_rd_idx_ps;
      rd_phase    <= rd_phase_ps;
      best_idx    <= best_idx_ps;
      best_byte   <= best_byte_ps;
      cur_bias    <= cur_bias_ps;
      sel_idx     <= sel_idx_ps;
      last_val    <= last_val_ps;
      last_idx    <= last_idx_ps;
      have_last   <= have_last_ps;
      fused_flow  <= fused_flow_ps;
      fetch_idx   <= fetch_idx_ps;
      comp_idx    <= comp_idx_ps;
      have_row    <= have_row_ps;
      inflight_idx <= inflight_idx_ps;
      req_inflight <= req_inflight_ps;
      psum    <= psum_ps;
      p_valid <= p_valid_ps;
      p_idx   <= p_idx_ps;
      act     <= act_ps;
      a_valid <= a_valid_ps;
      a_idx   <= a_idx_ps;
      in_vec      <= in_vec_ps;
      bias_buf    <= bias_buf_ps;
      wgt_vec     <= wgt_vec_ps;
      out_buf     <= out_buf_ps;
    end
  end

  //--------------------------------------------------------------------------------------------------------
  // Comb function : one fully-connected output element (flat 32-MAC dot product + bias, ReLU, descale /256,
  // saturate to 255). UNCHANGED from the pre-Select linear -> bit-exact vs the C golden lin_elem_nox.

  function automatic logic [7:0] calc_lin_element;
    input        [DIM_MAX_SIZE-1:0][7:0]  wgt;
    input signed [MAX_DOT_PROD_WIDTH-1:0] bias_in;
    input        [DIM_MAX_SIZE-1:0][7:0]  inv;

    logic signed [MAX_DOT_PROD_WIDTH-1:0] acc;
    logic signed [MAX_DOT_PROD_WIDTH-1:0] descaled_acc;
    logic signed [8:0]  in_val_s;
    logic signed [8:0]  wgt_val_s;
    logic signed [17:0] mult_val;
    integer i;

    begin
      acc = bias_in;
      for (i = 0; i < DIM_MAX_SIZE; i = i + 1) begin
        in_val_s  = {1'b0, inv[i]};            // input is uint8 -> keep positive
        wgt_val_s = {wgt[i][7], wgt[i]};       // weight is int8 -> sign-extend
        mult_val  = in_val_s * wgt_val_s;
        acc       = acc + mult_val;
      end
      if (acc < 0) acc = 0;                     // ReLU
      descaled_acc = acc >>> 8;                 // descale /256
      if (descaled_acc > 255) calc_lin_element = 8'd255;
      else                    calc_lin_element = descaled_acc[7:0];
    end
  endfunction

  //--------------------------------------------------------------------------------------------------------
  // Comb function (LIN_MACPIPE S2) : 32 signed 9x9 products grouped into 4 partial sums of 8 taps each.
  // The 8-tap sum is a BALANCED ADDER TREE (depth 3: (p0+p1)+(p2+p3) + (p4+p5)+(p6+p7)) instead of the
  // old 7-deep sequential accumulate -> the S2 critical cone (mult -> sum -> psum, the slrx worst path)
  // shrinks, raising Fmax. Bit-exact: two's-complement integer add is associative at ACC_W=32 with no
  // overflow (8 * 18-bit products < 2^21), so tree == sequential sum == calc_lin_element's Sum(32 taps).
  // Products are VERBATIM from calc_lin_element.
  function automatic logic [3:0][ACC_W-1:0] mac_group_partials;
    input [DIM_MAX_SIZE-1:0][7:0] wgt;
    input [DIM_MAX_SIZE-1:0][7:0] inv;
    logic signed [8:0]       in_val_s, wgt_val_s;
    logic signed [ACC_W-1:0] prod [0:7];                 // 8 sign-extended products of one group
    logic signed [ACC_W-1:0] s0, s1, s2, s3;             // tree level 1 (4 adds)
    logic signed [ACC_W-1:0] t0, t1;                     // tree level 2 (2 adds)
    integer g, k, i;
    begin
      for (g = 0; g < 4; g = g + 1) begin
        for (k = 0; k < 8; k = k + 1) begin
          i         = (g*8) + k;
          in_val_s  = {1'b0, inv[i]};                    // input uint8 -> zero-extend (verbatim)
          wgt_val_s = {wgt[i][7], wgt[i]};               // weight int8 -> sign-extend (verbatim)
          prod[k]   = $signed(in_val_s) * $signed(wgt_val_s);  // 18-bit product, sign-extended to ACC_W
        end
        s0 = prod[0] + prod[1];                          // level 1
        s1 = prod[2] + prod[3];
        s2 = prod[4] + prod[5];
        s3 = prod[6] + prod[7];
        t0 = s0 + s1;                                    // level 2
        t1 = s2 + s3;
        mac_group_partials[g] = t0 + t1;                 // level 3
      end
    end
  endfunction

endmodule
