/*
 * worker_pool.c - worker threads kept for the life of their owner. See
 * worker_pool.h.
 *
 * Why at all: the HEVC decoder's wavefront made fifteen threads and joined
 * them again for every picture, and so did the loop filters and the copy
 * into the surface. On a BC-250 fifteen creates and joins cost a quarter of
 * a millisecond, and a thread new to the processor starts with nothing in
 * its caches.
 */
#include "worker_pool.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdlib.h>

struct worker_pool {
    pthread_mutex_t m;
    pthread_cond_t go, finished;
    void *(*fn)(void *);
    void *arg;
    unsigned generation;            /* one more for every job */
    int want, taken, running;       /* of the current job's helpers */
    bool quit;
    int n;
    pthread_t thread[WORKER_POOL_MAX_THREADS];
};

static void *pool_worker(void *arg)
{
    worker_pool_t *p = arg;
    unsigned seen = 0;
    pthread_mutex_lock(&p->m);
    for (;;) {
        while (!p->quit && p->generation == seen)
            pthread_cond_wait(&p->go, &p->m);
        if (p->quit) break;
        seen = p->generation;
        /* A job wants a certain number of helpers; the rest of the pool
         * goes back to sleep. */
        if (p->taken >= p->want) continue;
        p->taken++;
        void *(*fn)(void *) = p->fn;
        void *a = p->arg;
        pthread_mutex_unlock(&p->m);
        fn(a);
        pthread_mutex_lock(&p->m);
        if (--p->running == 0) pthread_cond_signal(&p->finished);
    }
    pthread_mutex_unlock(&p->m);
    return NULL;
}

worker_pool_t *worker_pool_create(int threads)
{
    if (threads > WORKER_POOL_MAX_THREADS) threads = WORKER_POOL_MAX_THREADS;
    if (threads < 2) return NULL;

    worker_pool_t *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    pthread_mutex_init(&p->m, NULL);
    pthread_cond_init(&p->go, NULL);
    pthread_cond_init(&p->finished, NULL);
    for (int i = 0; i < threads - 1; i++)
        if (pthread_create(&p->thread[p->n], NULL, pool_worker, p) == 0)
            p->n++;
    if (!p->n) {
        worker_pool_destroy(p);
        return NULL;
    }
    return p;
}

int worker_pool_helpers(const worker_pool_t *p, int n)
{
    if (!p) return 0;
    const int h = n - 1;
    return h < 0 ? 0 : (h > p->n ? p->n : h);
}

void worker_pool_run(worker_pool_t *p, void *(*fn)(void *), void *arg, int n)
{
    const int helpers = worker_pool_helpers(p, n);
    if (helpers) {
        pthread_mutex_lock(&p->m);
        p->fn = fn;
        p->arg = arg;
        p->want = helpers;
        p->taken = 0;
        p->running = helpers;
        p->generation++;
        pthread_cond_broadcast(&p->go);
        pthread_mutex_unlock(&p->m);
    }
    fn(arg);                        /* this thread works too */
    if (helpers) {
        pthread_mutex_lock(&p->m);
        while (p->running > 0) pthread_cond_wait(&p->finished, &p->m);
        pthread_mutex_unlock(&p->m);
    }
}

typedef struct {
    atomic_int next;
    int count, chunk;
    void (*fn)(void *arg, int begin, int end);
    void *arg;
} range_job_t;

static void *range_worker(void *arg)
{
    range_job_t *j = arg;
    for (;;) {
        const int b = atomic_fetch_add_explicit(&j->next, j->chunk, memory_order_relaxed);
        if (b >= j->count) break;
        j->fn(j->arg, b, b + j->chunk < j->count ? b + j->chunk : j->count);
    }
    return NULL;
}

void worker_pool_for(worker_pool_t *p, int count, int chunk,
                     void (*fn)(void *arg, int begin, int end), void *arg, int n)
{
    if (count <= 0) return;
    if (chunk < 1) chunk = 1;
    if (!p || n < 2 || count <= chunk) {
        fn(arg, 0, count);
        return;
    }
    const int pieces = (count + chunk - 1) / chunk;
    range_job_t j = { .count = count, .chunk = chunk, .fn = fn, .arg = arg };
    atomic_init(&j.next, 0);
    worker_pool_run(p, range_worker, &j, n < pieces ? n : pieces);
}

void worker_pool_destroy(worker_pool_t *p)
{
    if (!p) return;
    pthread_mutex_lock(&p->m);
    p->quit = true;
    pthread_cond_broadcast(&p->go);
    pthread_mutex_unlock(&p->m);
    for (int i = 0; i < p->n; i++) pthread_join(p->thread[i], NULL);
    pthread_mutex_destroy(&p->m);
    pthread_cond_destroy(&p->go);
    pthread_cond_destroy(&p->finished);
    free(p);
}
