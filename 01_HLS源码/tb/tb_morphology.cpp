// ============================================================
// tb_morphology.cpp — 3x3 形态学 testbench (32bit TDATA + sideband)
//   模式A(默认): 合成图像 + 参考实现自检
//   模式B: 设 HLS_PGM_DIR=<目录> 时读 input.pgm, 输出 out_erode.pgm /
//          out_dilate.pgm 供 Python+OpenCV 比对
//   参考实现边界与 OpenCV 默认一致: 腐蚀补 255, 膨胀补 0
// ============================================================
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "pgm_io.h"

typedef ap_axiu<32, 1, 1, 1> px_t;

void morphology(hls::stream<px_t>& src, hls::stream<px_t>& dst,
                int mode, int width, int height, int enable);

#define TB_W 64
#define TB_H 48

static inline unsigned char chan(ap_uint<32> d) { return (unsigned char)d.range(23, 16).to_uint(); }

static void ref_morph(const unsigned char* in, unsigned char* out, int W, int H, int mode) {
    unsigned char border = (mode == 1) ? 255 : 0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            unsigned char acc = (mode == 1) ? 255 : 0;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++) {
                    int yy = y + dy, xx = x + dx;
                    unsigned char v = (yy < 0 || yy >= H || xx < 0 || xx >= W) ? border
                                                                               : in[yy * W + xx];
                    if (mode == 1) { if (v < acc) acc = v; }
                    else           { if (v > acc) acc = v; }
                }
            out[y * W + x] = acc;
        }
}

static int run_one(const unsigned char* in, unsigned char* out, int W, int H, int mode) {
    hls::stream<px_t> src("src");
    hls::stream<px_t> dst("dst");
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            unsigned char v = in[y * W + x];
            px_t p;
            p.data = ((ap_uint<32>)v << 16) | ((ap_uint<32>)v << 8) | (ap_uint<32>)v;
            p.user = (y == 0 && x == 0) ? 1 : 0;
            p.last = (x == W - 1) ? 1 : 0;
            p.keep = 0xF; p.strb = 0xF; p.id = 0; p.dest = 0;
            src.write(p);
        }

    morphology(src, dst, mode, W, H, 1);

    unsigned char* exp = new unsigned char[W * H];
    ref_morph(in, exp, W, H, mode);

    int mismatch = 0, maxdiff = 0, cnt = 0, side_bad = 0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            if (dst.empty()) { printf("错误: 输出流提前为空 (y=%d,x=%d, mode=%d)\n", y, x, mode); delete[] exp; return -1; }
            px_t o = dst.read();
            unsigned char got = chan(o.data);
            out[y * W + x] = got;
            int d = (int)got - (int)exp[y * W + x];
            if (d != 0) { mismatch++; if (abs(d) > maxdiff) maxdiff = abs(d); }
            cnt++;
            int eu = (y == 0 && x == 0) ? 1 : 0;
            int el = (y == H - 1 && x == W - 1) ? 1 : 0;   // TLAST 只在整帧最后像素
            if ((int)o.user.to_uint() != eu || (int)o.last.to_uint() != el) side_bad++;
        }
    delete[] exp;
    printf("mode=%s: 输出 %d 像素, 与参考不一致 = %d, 最大误差 = %d, sideband 异常 = %d\n",
           mode == 1 ? "erode " : "dilate", cnt, mismatch, maxdiff, side_bad);
    if (mismatch != 0 || side_bad != 0) return 1;
    return 0;
}

int main() {
    printf("========== tb_morphology (32bit + sideband) ==========\n");
    const char* pgm_dir = getenv("HLS_PGM_DIR");
    int W = TB_W, H = TB_H;
    unsigned char* in = 0;

    if (pgm_dir) {
        const char* fname = getenv("HLS_PGM_IN");
        if (!fname) fname = "input.pgm";
        char p[512];
        snprintf(p, sizeof(p), "%s/%s", pgm_dir, fname);
        in = pgm_read(p, W, H);
        printf("读取输入文件: %s\n", p);
    }
    if (!in) {
        W = TB_W; H = TB_H;
        in = new unsigned char[W * H];
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                unsigned char v = 0;
                if ((x / 8 + y / 8) % 2 == 0) v = 255;
                if (x > W / 4 && x < 3 * W / 4 && y > H / 4 && y < 3 * H / 4) v = 255;
                if (x > W / 2 && y < H / 4) v = 128;
                in[y * W + x] = v;
            }
        printf("使用合成测试图 %dx%d\n", W, H);
    } else {
        printf("使用文件测试图 %dx%d\n", W, H);
    }

    unsigned char* out_e = new unsigned char[W * H];
    unsigned char* out_d = new unsigned char[W * H];
    int bad_e = run_one(in, out_e, W, H, 1);
    int bad_d = run_one(in, out_d, W, H, 2);

    if (pgm_dir) {
        char p[512];
        snprintf(p, sizeof(p), "%s/out_erode.pgm", pgm_dir);  pgm_write(p, out_e, W, H);
        snprintf(p, sizeof(p), "%s/out_dilate.pgm", pgm_dir); pgm_write(p, out_d, W, H);
    }

    delete[] in; delete[] out_e; delete[] out_d;
    if (bad_e == 0 && bad_d == 0) { printf(">>> PASS\n"); return 0; }
    printf(">>> FAIL\n");
    return 1;
}
