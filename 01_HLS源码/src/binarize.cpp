// ============================================================
// binarize.cpp — 图像二值化 HLS 算子 (32bit AXI4-Stream + sideband)
//
//   1) 调用一次 = 处理一帧, 像素数 = width*height (寄存器配置)
//   2) 流格式: ap_axiu<32,1,1,1>  —— TDATA 32bit(低 24bit = RGB)
//      带 TLAST/TKEEP/TSTRB 等 sideband, 与 AXI DMA 兼容性最好
//   3) 灰度公式与 OpenCV COLOR_RGB2GRAY 完全一致(定点)
//   4) enable=0 时直通旁路
// ============================================================
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <ap_int.h>

typedef ap_axiu<32, 1, 1, 1> px_t;

static ap_uint<8> rgb2gray(ap_uint<32> d) {
#pragma HLS INLINE
    ap_uint<8>  r = d.range(23, 16);
    ap_uint<8>  g = d.range(15, 8);
    ap_uint<8>  b = d.range(7, 0);
    ap_uint<32> y = (ap_uint<32>)4899 * r + (ap_uint<32>)9617 * g + (ap_uint<32>)1868 * b;
    return (ap_uint<8>)((y + 8192) >> 14);
}

void binarize(hls::stream<px_t>& src, hls::stream<px_t>& dst,
              int thresh, int width, int height, int enable) {
#pragma HLS INTERFACE axis      port=src
#pragma HLS INTERFACE axis      port=dst
#pragma HLS INTERFACE s_axilite port=thresh bundle=control
#pragma HLS INTERFACE s_axilite port=width  bundle=control
#pragma HLS INTERFACE s_axilite port=height bundle=control
#pragma HLS INTERFACE s_axilite port=enable bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int total = width * height;

    for (int i = 0; i < total; i++) {
#pragma HLS PIPELINE II=1
        px_t p = src.read();
        if (enable != 0) {
            ap_uint<8> gray = rgb2gray(p.data);
            p.data = (gray > (ap_uint<8>)thresh) ? (ap_uint<32>)0x00FFFFFF
                                                 : (ap_uint<32>)0x00000000;
        }
        dst.write(p);          // sideband(user/last/keep/strb) 原样透传
    }
}
