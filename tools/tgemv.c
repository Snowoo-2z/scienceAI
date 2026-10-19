/*
 * tgemv.c — débit réel d'un produit matrice-vecteur ternaire (2 bits packés).
 *
 * Auteur : KAIROS · 2026-10-19 · domaine public (CC0)
 *
 * Pourquoi ce benchmark existe
 * ----------------------------
 * Tout le régime « ~100 Go » repose sur une hypothèse : des experts à 1,58 bit
 * sont limités par la bande passante mémoire, pas par le calcul. C'est vrai
 * seulement si le noyau de déquantification+produit scalaire tient la cadence.
 *
 * Si le noyau ne fait que 200 GOP/s, un modèle à 6 G paramètres actifs coûte
 * 12 GFLOP/token → 60 ms/token → 17 tok/s, même avec une bande passante
 * infinie. Le noyau est donc le vrai plafond, et il faut le mesurer.
 *
 * Trois implémentations sont comparées :
 *   A. naïve      : décodage 2 bits + produit scalaire, un poids à la fois ;
 *   B. table      : table de 256 entrées donnant les 4 poids d'un octet ;
 *   C. déballage  : on déballe en int8 dans un tampon, puis dot product
 *                   (que le compilateur peut vectoriser).
 *
 * Compilation
 * -----------
 *   gcc -O3 -march=native -o tgemv tools/tgemv.c
 *   ./tgemv            # taille d'un expert HECATON (2048 x 1024)
 *   ./tgemv 4096 4096  # autre géométrie
 *
 * Lecture des résultats
 * ---------------------
 *   GOP/s = 2 * M * K / temps.  C'est le débit de MAC (multiply-accumulate)
 *   équivalent. Un noyau AVX-512 VNNI bien écrit fait 4 à 10x mieux que le
 *   mode C sur la même machine ; les chiffres ici sont un plancher, pas un
 *   plafond. Ce qui compte : le rapport au seuil mémoire.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* 0 -> 0, 1 -> +1, 2 -> -1 (encodage trinaire compact) */
static const int8_t TERN[4] = {0, 1, -1, 0};

/* ---------------------------------------------------------------- */
/* A. naïve                                                          */
/* ---------------------------------------------------------------- */
static void gemv_naive(const uint8_t *W, const int8_t *x, int32_t *y,
                       int M, int K, int passes) {
    const int KB = K / 4;
    for (int p = 0; p < passes; p++) {
        for (int r = 0; r < M; r++) {
            const uint8_t *row = W + (size_t)r * KB;
            int32_t acc = 0;
            for (int i = 0; i < KB; i++) {
                uint8_t b = row[i];
                const int8_t *xx = x + 4 * i;
                acc += TERN[b & 3] * xx[0];
                acc += TERN[(b >> 2) & 3] * xx[1];
                acc += TERN[(b >> 4) & 3] * xx[2];
                acc += TERN[(b >> 6) & 3] * xx[3];
            }
            y[r] = acc;
        }
    }
}

/* ---------------------------------------------------------------- */
/* B. table de 256 entrées : 4 poids décodés d'un coup               */
/* ---------------------------------------------------------------- */
static int8_t LUT[256][4];

static void build_lut(void) {
    for (int b = 0; b < 256; b++) {
        LUT[b][0] = TERN[b & 3];
        LUT[b][1] = TERN[(b >> 2) & 3];
        LUT[b][2] = TERN[(b >> 4) & 3];
        LUT[b][3] = TERN[(b >> 6) & 3];
    }
}

static void gemv_lut(const uint8_t *W, const int8_t *x, int32_t *y,
                     int M, int K, int passes) {
    const int KB = K / 4;
    for (int p = 0; p < passes; p++) {
        for (int r = 0; r < M; r++) {
            const uint8_t *row = W + (size_t)r * KB;
            const int8_t *xx = x;
            int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            int i = 0;
            for (; i + 4 <= KB; i += 4) {
                const int8_t *t0 = LUT[row[i + 0]];
                const int8_t *t1 = LUT[row[i + 1]];
                const int8_t *t2 = LUT[row[i + 2]];
                const int8_t *t3 = LUT[row[i + 3]];
                a0 += t0[0]*xx[0] + t0[1]*xx[1] + t0[2]*xx[2] + t0[3]*xx[3];
                a1 += t1[0]*xx[4] + t1[1]*xx[5] + t1[2]*xx[6] + t1[3]*xx[7];
                a2 += t2[0]*xx[8] + t2[1]*xx[9] + t2[2]*xx[10] + t2[3]*xx[11];
                a3 += t3[0]*xx[12]+ t3[1]*xx[13]+ t3[2]*xx[14]+ t3[3]*xx[15];
                xx += 16;
            }
            for (; i < KB; i++) {
                a0 += LUT[row[i]][0]*xx[0] + LUT[row[i]][1]*xx[1]
                    + LUT[row[i]][2]*xx[2] + LUT[row[i]][3]*xx[3];
                xx += 4;
            }
            y[r] = a0 + a1 + a2 + a3;
        }
    }
}

/* ---------------------------------------------------------------- */
/* C. déballage en int8 puis dot product                             */
/* ---------------------------------------------------------------- */
static void gemv_unpack(const uint8_t *W, const int8_t *x, int32_t *y,
                        int M, int K, int passes, int8_t *tmp) {
    const int KB = K / 4;
    for (int p = 0; p < passes; p++) {
        for (int r = 0; r < M; r++) {
            const uint8_t *row = W + (size_t)r * KB;
            for (int i = 0; i < KB; i++) {
                const int8_t *t = LUT[row[i]];
                tmp[4*i+0] = t[0]; tmp[4*i+1] = t[1];
                tmp[4*i+2] = t[2]; tmp[4*i+3] = t[3];
            }
            int32_t acc = 0;
            for (int i = 0; i < K; i++) acc += tmp[i] * x[i];
            y[r] = acc;
        }
    }
}

/* ---------------------------------------------------------------- */

int main(int argc, char **argv) {
    int M = 2048, K = 1024;          /* une matrice d'expert HECATON */
    if (argc >= 3) { M = atoi(argv[1]); K = atoi(argv[2]); }
    if (K % 4) { fprintf(stderr, "K doit être multiple de 4\n"); return 1; }

    int passes = 50;
    if (argc >= 4) passes = atoi(argv[3]);

    size_t nw = (size_t)M * K;
    size_t nbytes = (nw + 3) / 4;
    uint8_t *W = malloc(nbytes);
    int8_t *x = malloc((size_t)K);
    int8_t *tmp = malloc((size_t)K);
    int32_t *y = malloc(sizeof(int32_t) * (size_t)M);
    if (!W || !x || !tmp || !y) { fprintf(stderr, "malloc\n"); return 1; }

    srand(12345);
    for (size_t i = 0; i < nbytes; i++) W[i] = (uint8_t)(rand() & 0xFF);
    for (int i = 0; i < K; i++) x[i] = (int8_t)((rand() % 255) - 127);
    build_lut();

    double ops = 2.0 * (double)M * (double)K * (double)passes;

    double t0 = now();
    gemv_naive(W, x, y, M, K, passes);
    double tA = now() - t0;

    t0 = now();
    gemv_lut(W, x, y, M, K, passes);
    double tB = now() - t0;

    t0 = now();
    gemv_unpack(W, x, y, M, K, passes, tmp);
    double tC = now() - t0;

    double gopsA = ops / tA / 1e9;
    double gopsB = ops / tB / 1e9;
    double gopsC = ops / tC / 1e9;

    printf("tgemv — produit matrice-vecteur ternaire (2 bits packés)\n");
    printf("matrice %d x %d · %d passes · %.1f Mo de poids\n",
           M, K, passes, (double)nbytes / 1e6);
    printf("------------------------------------------------------------------\n");
    printf("  A. naïve (1 poids/fois) ...... %6.2f GOP/s\n", gopsA);
    printf("  B. table 256 entrées ......... %6.2f GOP/s  (x%.1f vs A)\n", gopsB, gopsB / gopsA);
    printf("  C. déballage + dot product ... %6.2f GOP/s  (x%.1f vs A)\n", gopsC, gopsC / gopsA);
    printf("\n");

    /* seuil mémoire : à partir de quelle bande passante le noyau devient-il
     * le goulot ? temps_calcul = 2*P/GOPS ; temps_memoire = P*b/8/BW */
    double bits = 1.7, gops = gopsC;
    printf("  Pour un modèle à 1.7 bit/poids, le noyau C est le goulot\n");
    printf("  en dessous de %.0f Go/s de bande passante utile.\n", gops * bits / 16.0);
    printf("  (seuil = GOPS * bits / 16 ; au-dessus, on est mémoire-borné)\n");
    printf("\n");
    printf("  Attention : 2 vCPU seulement ici. Un noyau AVX-512 VNNI\n");
    printf("  écrit à la main fait 4 à 10x mieux. Comparer les RATIOS.\n");

    free(W); free(x); free(tmp); free(y);
    return 0;
}
