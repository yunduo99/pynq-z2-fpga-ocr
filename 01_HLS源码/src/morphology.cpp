// ============================================================
// morphology.cpp — 3x3 形态学滤波 HLS 算子 (32bit AXI4-Stream + sideband)
//   1) 模式: 0=直通  1=腐蚀(erode)  2=膨胀(dilate)
//   2) 边界: 与 OpenCV 默认一致 (腐蚀补 255, 膨胀补 0)
//   3) 流格式: ap_axiu<32,1,1,1>, TDATA 低 24bit = RGB, 取 R 通道算强度
//   4) 结构: 3 行缓存轮转 + 3 列移位寄存器 -> 3x3 窗口, II=1
// ============================================================
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <ap_int.h>

typedef ap_axiu<32, 1, 1, 1> px_t;

#define MAX_W 1920

static ap_uint<8> get_intensity(ap_uint<32> d) {
#pragma HLS INLINE
    return d.range(23, 16);
}

void morphology(hls::stream<px_t>& src, hls::stream<px_t>& dst,
                int mode, int width, int height, int enable) {
#pragma HLS INTERFACE axis      port=src
#pragma HLS INTERFACE axis      port=dst
#pragma HLS INTERFACE s_axilite port=mode   bundle=control
#pragma HLS INTERFACE s_axilite port=width  bundle=control
#pragma HLS INTERFACE s_axilite port=height bundle=control
#pragma HLS INTERFACE s_axilite port=enable bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static ap_uint<8> lb[3][MAX_W];
#pragma HLS ARRAY_PARTITION variable=lb dim=1 complete

    const int W = width;
    const int H = height;

    // ---- 模式 0 / 关闭: 直通 ----
    if (mode == 0 || enable == 0) {
        for (int i = 0; i < W * H; i++) {
#pragma HLS PIPELINE II=1
            dst.write(src.read());
        }
        return;
    }

    const ap_uint<8> border = (mode == 1) ? (ap_uint<8>)255 : (ap_uint<8>)0;

    ap_uint<8> c_m2[3], c_m1[3], c_cu[3];
#pragma HLS ARRAY_PARTITION variable=c_m2 complete
#pragma HLS ARRAY_PARTITION variable=c_m1 complete
#pragma HLS ARRAY_PARTITION variable=c_cu complete

    ap_uint<8> win[3][3];
#pragma HLS ARRAY_PARTITION variable=win complete

    for (int y = 0; y <= H; y++) {
        const int s_new = y % 3;
        const int s_m1  = (y + 2) % 3;
        const int s_m2  = (y + 1) % 3;

        c_m2[0] = border; c_m2[1] = border; c_m2[2] = border;
        c_m1[0] = border; c_m1[1] = border; c_m1[2] = border;
        c_cu[0] = border; c_cu[1] = border; c_cu[2] = border;

        for (int x = 0; x <= W; x++) {
#pragma HLS PIPELINE II=1
            ap_uint<8> v_new = border;
            if (x < W && y < H) {
                px_t p = src.read();
                v_new = get_intensity(p.data);
            }

            ap_uint<8> v_m1 = border;
            ap_uint<8> v_m2 = border;
            if (x < W) {
                if (y >= 1) v_m1 = lb[s_m1][x];
                if (y >= 2) v_m2 = lb[s_m2][x];
            }
            if (x < W && y < H) lb[s_new][x] = v_new;

            c_m2[0] = c_m2[1]; c_m2[1] = c_m2[2]; c_m2[2] = v_m2;
            c_m1[0] = c_m1[1]; c_m1[1] = c_m1[2]; c_m1[2] = v_m1;
            c_cu[0] = c_cu[1]; c_cu[1] = c_cu[2]; c_cu[2] = v_new;

            if (y >= 1 && x >= 1) {
                win[0][0] = c_m2[0]; win[0][1] = c_m2[1]; win[0][2] = c_m2[2];
                win[1][0] = c_m1[0]; win[1][1] = c_m1[1]; win[1][2] = c_m1[2];
                win[2][0] = c_cu[0]; win[2][1] = c_cu[1]; win[2][2] = c_cu[2];

                ap_uint<8> res = win[0][0];
                if (mode == 1) {
                    for (int r = 0; r < 3; r++)
                        for (int c = 0; c < 3; c++)
                            if (win[r][c] < res) res = win[r][c];
                } else {
                    for (int r = 0; r < 3; r++)
                        for (int c = 0; c < 3; c++)
                            if (win[r][c] > res) res = win[r][c];
                }

                px_t o;
                o.data = ((ap_uint<32>)res << 16) | ((ap_uint<32>)res << 8) | (ap_uint<32>)res;
                o.user = (y == 1 && x == 1) ? 1 : 0;      // SOF: 整帧第一个输出像素
                o.last = (y == H && x == W) ? 1 : 0;      // EOL: 只在整帧最后一个像素拉高!
                o.keep = 0xF; o.strb = 0xF; o.id = 0; o.dest = 0;
                dst.write(o);
            }
        }
    }
}
