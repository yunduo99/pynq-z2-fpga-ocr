// ============================================================
// tb_binarize.cpp — 二值化算子 testbench (32bit TDATA + sideband)
//   模式A(默认): 合成图像 + 参考实现自检
//   模式B: 设 HLS_PGM_DIR=<目录> 时读 input.pgm/threshold.txt,
//          写出 out_bin.pgm 供 Python+OpenCV 逐像素比对
// ============================================================
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "pgm_io.h"

typedef ap_axiu<32, 1, 1, 1> px_t;

void binarize(hls::stream<px_t>& src, hls::stream<px_t>& dst,
              int thresh, int width, int height, int enable);

#define TB_W 64
#define TB_H 48

static inline unsigned char chan(ap_uint<32> d) { return (unsigned char)d.range(23, 16).to_uint(); }

static unsigned char ref_gray(int r, int g, int b) {
    unsigned int y = 4899u * r + 9617u * g + 1868u * b;
    return (unsigned char)((y + 8192u) >> 14);
}

int main() {
    printf("========== tb_binarize (32bit + sideband) ==========\n");
    const char* pgm_dir = getenv("HLS_PGM_DIR");
    int W = TB_W, H = TB_H;
    unsigned char* in_img = 0;
    int thresh = 128;
    bool from_file = false;

    if (pgm_dir) {
        char p[512];
        snprintf(p, sizeof(p), "%s/input.pgm", pgm_dir);
        in_img = pgm_read(p, W, H);
        if (in_img) {
            from_file = true;
            snprintf(p, sizeof(p), "%s/threshold.txt", pgm_dir);
            FILE* fp = fopen(p, "r");
            if (fp) { if (fscanf(fp, "%d", &thresh) != 1) thresh = 128; fclose(fp); }
        }
    }
    if (!in_img) {
        W = TB_W; H = TB_H;
        in_img = new unsigned char[W * H];
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                int v = (x * 4) % 256;
                if (y > H / 3 && y < 2 * H / 3) v = 90;
                in_img[y * W + x] = (unsigned char)v;
            }
        printf("使用合成测试图 %dx%d\n", W, H);
    } else {
        printf("使用文件测试图 %dx%d\n", W, H);
    }
    printf("thresh = %d%s\n", thresh, from_file ? " (来自 threshold.txt)" : "");

    hls::stream<px_t> src("src");
    hls::stream<px_t> dst("dst");
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            unsigned char v = in_img[y * W + x];
            px_t p;
            p.data = ((ap_uint<32>)v << 16) | ((ap_uint<32>)v << 8) | (ap_uint<32>)v;
            p.user = (y == 0 && x == 0) ? 1 : 0;
            p.last = (x == W - 1) ? 1 : 0;
            p.keep = 0xF; p.strb = 0xF; p.id = 0; p.dest = 0;
            src.write(p);
        }
    printf("已注入 %d 个像素\n", W * H);

    binarize(src, dst, thresh, W, H, 1);

    unsigned char* out = new unsigned char[W * H];
    int mismatch = 0, maxdiff = 0, side_bad = 0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            if (dst.empty()) { printf("错误: 输出流提前为空 (y=%d,x=%d)\n", y, x); return 1; }
            px_t o = dst.read();
            unsigned char got = chan(o.data);
            unsigned char v   = in_img[y * W + x];
            unsigned char exp = (ref_gray(v, v, v) > (unsigned char)thresh) ? 255 : 0;
            out[y * W + x] = got;
            int d = (int)got - (int)exp;
            if (d != 0) { mismatch++; if (abs(d) > maxdiff) maxdiff = abs(d); }
            if ((int)o.last.to_uint() != (x == W - 1 ? 1 : 0)) side_bad++;
        }
    printf("比对结果: 不一致像素 = %d / %d, 最大误差 = %d, sideband 异常 = %d\n",
           mismatch, W * H, maxdiff, side_bad);

    // 旁路模式
    hls::stream<px_t> s2("s2");
    hls::stream<px_t> d2("d2");
    for (int i = 0; i < W * H; i++) {
        px_t p; unsigned char v = in_img[i];
        p.data = ((ap_uint<32>)v << 16) | ((ap_uint<32>)v << 8) | (ap_uint<32>)v;
        p.user = 0; p.last = 0; p.keep = 0xF; p.strb = 0xF; p.id = 0; p.dest = 0;
        s2.write(p);
    }
    binarize(s2, d2, thresh, W, H, 0);
    int bypass_bad = 0;
    for (int i = 0; i < W * H; i++) {
        px_t o = d2.read();
        if (chan(o.data) != in_img[i]) bypass_bad++;
    }
    printf("旁路模式(enable=0): 不一致 = %d (应为 0)\n", bypass_bad);

    if (pgm_dir) {
        char p[512];
        snprintf(p, sizeof(p), "%s/out_bin.pgm", pgm_dir);
        pgm_write(p, out, W, H);
    }

    delete[] in_img; delete[] out;
    if (mismatch == 0 && bypass_bad == 0 && side_bad == 0) { printf(">>> PASS\n"); return 0; }
    printf(">>> FAIL\n");
    return 1;
}
