// ============================================================
// pgm_io.h — 简易 P5 (二进制) PGM 读写, 零外部依赖
// 用于 HLS testbench 与 OpenCV 生成的参考图做逐像素比对
// ============================================================
#ifndef PGM_IO_H
#define PGM_IO_H

#include <cstdio>
#include <cstdlib>
#include <cstring>

// 读取 P5 PGM; 成功返回 new 出来的缓冲(调用方 delete[]), 失败返回 0
inline unsigned char* pgm_read(const char* path, int& w, int& h) {
    FILE* fp = fopen(path, "rb");
    if (!fp) { printf("[pgm] 打不开 %s\n", path); return 0; }
    char magic[3] = {0};
    if (fscanf(fp, "%2s", magic) != 1 || strcmp(magic, "P5") != 0) {
        printf("[pgm] 不是 P5 格式: %s\n", path); fclose(fp); return 0;
    }
    int maxv = 0;
    if (fscanf(fp, "%d %d %d", &w, &h, &maxv) != 3) {
        printf("[pgm] 头部解析失败\n"); fclose(fp); return 0;
    }
    fgetc(fp);                                   // 头部后的单个空白
    unsigned char* buf = new unsigned char[w * h];
    size_t n = fread(buf, 1, (size_t)w * h, fp);
    fclose(fp);
    if (n != (size_t)w * h) printf("[pgm] 警告: 只读到 %zu / %d 字节\n", n, w * h);
    return buf;
}

// 写出 P5 PGM
inline void pgm_write(const char* path, const unsigned char* buf, int w, int h) {
    FILE* fp = fopen(path, "wb");
    if (!fp) { printf("[pgm] 无法写入 %s\n", path); return; }
    fprintf(fp, "P5\n%d %d\n255\n", w, h);
    fwrite(buf, 1, (size_t)w * h, fp);
    fclose(fp);
    printf("[pgm] 已写出 %s (%dx%d)\n", path, w, h);
}

#endif
