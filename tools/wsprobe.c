/* wsprobe.c — mesure du chargement de « cartouches » depuis le disque.
 *
 * Question : si un modèle garde son cœur en RAM et charge une cartouche de
 * compétence par requête, combien de temps perd-on ? Et si les experts sont
 * paginés par token, quel débit disque peut-on espérer ?
 *
 * Le programme mesure :
 *   1. la lecture séquentielle (borne haute) ;
 *   2. la lecture aléatoire de blocs de 256 Ko à 8 Mo (le régime cartouche) ;
 *   3. le temps de chargement de 1, 2, 4, 8 Mo ;
 *   4. l'effet du cache page (option --no-fadvise pour le laisser actif).
 *
 * Build : gcc -O3 -o wsprobe tools/wsprobe.c
 * Run   : ./wsprobe --file /tmp/wsprobe.bin --size-mb 512
 *
 * Sortie : texte, une ligne par mesure, avec les six champs du protocole
 * (stockage · bloc · cache · syscall · matériel · date).
 */

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;

static uint64_t xorshift64(void) {
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng_state = x;
    return x;
}

static int make_file(const char *path, size_t size_bytes) {
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { perror("open create"); return -1; }
    if (ftruncate(fd, (off_t)size_bytes) != 0) { perror("ftruncate"); close(fd); return -1; }
    /* Écrire des données non nulles pour éviter un fichier sparse. */
    const size_t chunk = 1u << 20;
    char *buf = malloc(chunk);
    if (!buf) { close(fd); return -1; }
    for (size_t i = 0; i < chunk; i++) buf[i] = (char)(i * 131u + 17u);
    for (size_t off = 0; off < size_bytes; off += chunk) {
        size_t n = size_bytes - off < chunk ? size_bytes - off : chunk;
        ssize_t w = pwrite(fd, buf, n, (off_t)off);
        if (w != (ssize_t)n) { perror("pwrite"); free(buf); close(fd); return -1; }
    }
    fsync(fd);
    free(buf);
    return fd;
}

static double read_block(int fd, void *buf, size_t block, off_t off, int use_direct) {
    double t0 = now_s();
    ssize_t r;
    if (use_direct) {
        r = pread(fd, buf, block, off);
    } else {
        r = pread(fd, buf, block, off);
    }
    double t1 = now_s();
    if (r != (ssize_t)block) {
        /* Fin de fichier ou erreur : on renvoie -1 pour signaler. */
        return -1.0;
    }
    return t1 - t0;
}

static void evict(int fd, int enabled) {
    if (!enabled) return;
    /* Demande au noyau d'oublier les pages du fichier. Peut être partiel. */
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
}

int main(int argc, char **argv) {
    const char *path = "/tmp/wsprobe.bin";
    size_t size_mb = 512;
    int fadvise = 1;
    int direct = 0;
    int n_random = 64;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--file") && i + 1 < argc) path = argv[++i];
        else if (!strcmp(argv[i], "--size-mb") && i + 1 < argc) size_mb = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--no-fadvise")) fadvise = 0;
        else if (!strcmp(argv[i], "--direct")) direct = 1;
        else if (!strcmp(argv[i], "--n") && i + 1 < argc) n_random = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--help")) {
            printf("usage: %s [--file PATH] [--size-mb N] [--no-fadvise] [--direct] [--n N]\n", argv[0]);
            return 0;
        }
    }

    size_t size_bytes = size_mb * 1000000ull;
    int fd = make_file(path, size_bytes);
    if (fd < 0) return 1;

    if (direct) {
        /* Rouvrir en O_DIRECT ; si le FS ne le supporte pas, on continue en buffered. */
        int fd2 = open(path, O_RDONLY | O_DIRECT);
        if (fd2 >= 0) { close(fd); fd = fd2; }
        else { fprintf(stderr, "O_DIRECT indisponible (%s), lecture bufferisée\n", strerror(errno)); direct = 0; }
    } else {
        close(fd);
        fd = open(path, O_RDONLY);
        if (fd < 0) { perror("open read"); return 1; }
    }

    const size_t max_block = 8u << 20;
    char *buf = NULL;
    if (direct) {
        if (posix_memalign((void **)&buf, 4096, max_block) != 0) return 1;
    } else {
        buf = malloc(max_block);
    }
    if (!buf) return 1;

    printf("# wsprobe v1\n");
    printf("# date=2026-10-19\n");
    printf("# file=%s size_mb=%zu\n", path, size_mb);
    printf("# cache_control=%s\n", fadvise ? "posix_fadvise(DONTNEED)" : "page_cache_active");
    printf("# syscall=%s\n", direct ? "pread(O_DIRECT)" : "pread(buffered)");
    printf("# hw=%s\n", "Intel Xeon 2.60GHz, 2 vCPU, 4 Go RAM, ext4 /dev/root");
    printf("# protocole=stockage · bloc · cache · syscall · materiel · date\n");
    fflush(stdout);

    /* 1. Séquentiel, blocs de 8 Mo. */
    evict(fd, fadvise);
    {
        size_t block = max_block;
        size_t n = size_bytes / block;
        if (n > 64) n = 64;
        double t0 = now_s();
        size_t ok = 0;
        for (size_t i = 0; i < n; i++) {
            if (read_block(fd, buf, block, (off_t)(i * block), direct) >= 0) ok++;
        }
        double dt = now_s() - t0;
        double bw = (double)ok * block / dt / 1e6;
        printf("sequential block_mb=%.2f n=%zu bw_mbs=%.1f\n", block / 1e6, ok, bw);
    }

    /* 2. Aléatoire, tailles de bloc (puissances de deux, alignées 4096). */
    const size_t sizes[] = {256u << 10, 512u << 10, 1u << 20, 2u << 20, 4u << 20, 8u << 20};
    const int nsizes = (int)(sizeof(sizes) / sizeof(sizes[0]));
    for (int s = 0; s < nsizes; s++) {
        size_t block = sizes[s];
        if (block > max_block) continue;
        evict(fd, fadvise);
        double total = 0.0;
        size_t ok = 0;
        for (int i = 0; i < n_random; i++) {
            uint64_t r = xorshift64();
            size_t blocks = size_bytes / block;
            if (blocks == 0) break;
            off_t off = (off_t)((r % blocks) * block);
            double dt = read_block(fd, buf, block, off, direct);
            if (dt >= 0.0) { total += dt; ok++; }
        }
        double mean_ms = ok ? total / ok * 1000.0 : -1.0;
        double bw = (ok && total > 0) ? (double)ok * block / total / 1e6 : 0.0;
        printf("random block_kib=%zu n=%zu latency_ms=%.2f bw_mbs=%.1f\n",
               block >> 10, ok, mean_ms, bw);
    }

    /* 3. Chargement d'une cartouche : 16 tirages, mediane + moyenne. */
    printf("# cartridge_load (16 tirages independants, cache evince a chaque tirage)\n");
    const size_t cart[] = {256u << 10, 512u << 10, 1u << 20, 2u << 20, 4u << 20, 8u << 20};
    for (int i = 0; i < 6; i++) {
        size_t block = cart[i];
        if (block > max_block) continue;
        double vals[16];
        int ok = 0;
        for (int t = 0; t < 16; t++) {
            evict(fd, fadvise);
            uint64_t r = xorshift64();
            size_t blocks = size_bytes / block;
            if (blocks == 0) break;
            off_t off = (off_t)((r % blocks) * block);
            double dt = read_block(fd, buf, block, off, direct);
            if (dt >= 0.0) vals[ok++] = dt * 1000.0;
        }
        if (ok == 0) { printf("cartridge size_kib=%zu load_ms=NA\n", block >> 10); continue; }
        for (int a = 0; a < ok; a++) for (int b = a + 1; b < ok; b++)
            if (vals[b] < vals[a]) { double tmp = vals[a]; vals[a] = vals[b]; vals[b] = tmp; }
        double sum = 0.0;
        for (int a = 0; a < ok; a++) sum += vals[a];
        printf("cartridge size_kib=%zu n=%d load_ms_med=%.2f load_ms_mean=%.2f load_ms_min=%.2f load_ms_max=%.2f\n",
               block >> 10, ok, vals[ok / 2], sum / ok, vals[0], vals[ok - 1]);
    }

    free(buf);
    close(fd);
    return 0;
}
