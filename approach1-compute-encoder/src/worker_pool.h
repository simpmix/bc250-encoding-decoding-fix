/*
 * worker_pool.h - worker threads kept for the life of whatever owns them.
 *
 * ⚠️ One pool per owner - a decoder, an encoder, a GPU context - never one
 * per process. The driver is a library libva loads and unloads, and threads
 * left running in a library that has been unloaded take the process down the
 * next time they wake. A pool's threads are joined when it is destroyed.
 *
 * Its threads sleep on a condition variable between jobs. OpenMP's spin
 * instead, for a while after every parallel region, and on a machine also
 * running the game being streamed those are cycles taken from the game.
 */
#ifndef WORKER_POOL_H
#define WORKER_POOL_H

#define WORKER_POOL_MAX_THREADS 16

typedef struct worker_pool worker_pool_t;

/* A pool for jobs of up to `threads` threads: the caller's own and
 * threads - 1 helpers, at most WORKER_POOL_MAX_THREADS in all. NULL when
 * that is fewer than two or no helper could be started; every function
 * below takes NULL and then does the work on the caller alone. */
worker_pool_t *worker_pool_create(int threads);

/* How many helpers a job of n threads would get. */
int worker_pool_helpers(const worker_pool_t *p, int n);

/* fn(arg) on the caller and on up to n - 1 helpers; returns once every one
 * of them has returned. Not reentrant: one job at a time per pool. */
void worker_pool_run(worker_pool_t *p, void *(*fn)(void *), void *arg, int n);

/* fn(arg, begin, end) over [0, count) in pieces of `chunk`, handed out in
 * increasing order to up to n threads. */
void worker_pool_for(worker_pool_t *p, int count, int chunk,
                     void (*fn)(void *arg, int begin, int end), void *arg, int n);

void worker_pool_destroy(worker_pool_t *p);

#endif /* WORKER_POOL_H */
