// Local descriptors for nn::socket on a host with no network: event file descriptors and the calls that wait on them.
// Everything else in nn::socket still fails (ov_system.c, ov_socket_fail). See README, "Host event descriptors".
#include <stdint.h>
#include <string.h>

#include "host.h"
#include "nvn_api.h"
#include "ov.h"
#include "platform/platform.h"

enum { MAX_FDS = 32, FD_BASE = 64 };
enum { EFD_SEMAPHORE = 0x1, EFD_NONBLOCK = 0x800 };
enum { P_IN = 0x1, P_OUT = 0x4, P_NVAL = 0x20 };

typedef struct { int used, nonblock, semaphore; uint64_t counter; } EventFd;

static EventFd efds[MAX_FDS];
static PlatMutex lock = PLAT_MUTEX_INIT;

static EventFd *find(int fd) {
    if (fd < FD_BASE || fd >= FD_BASE + MAX_FDS || !efds[fd - FD_BASE].used) return NULL;
    return &efds[fd - FD_BASE];
}

// int EventFd(uint32_t initval, EventFdFlags flags)
static void ov_eventfd(Cpu *c) {
    uint64_t init = arg_x(c, 0);
    uint32_t flags = (uint32_t)arg_x(c, 1);
    int fd = -1;
    plat_mutex_lock(&lock);
    for (int i = 0; i < MAX_FDS; i++)
        if (!efds[i].used) {
            efds[i] = (EventFd){1, !!(flags & EFD_NONBLOCK), !!(flags & EFD_SEMAPHORE), init};
            fd = FD_BASE + i;
            break;
        }
    plat_mutex_unlock(&lock);
    ret_x(c, (uint64_t)(int64_t)fd);
}

// ssize_t Read(int fd, void *buf, size_t n)
static void ov_eventfd_read(Cpu *c) {
    int fd = (int)arg_x(c, 0);
    uint64_t *buf = arg_p(c, 1);
    uint64_t n = arg_x(c, 2);
    int64_t r = -1;
    for (;;) {
        plat_mutex_lock(&lock);
        EventFd *e = find(fd);
        if (!e || !buf || n < 8) { plat_mutex_unlock(&lock); break; }
        if (e->counter) {
            if (e->semaphore) *buf = 1, e->counter--;
            else *buf = e->counter, e->counter = 0;
            plat_mutex_unlock(&lock);
            r = 8;
            break;
        }
        int nonblock = e->nonblock;
        plat_mutex_unlock(&lock);
        if (nonblock) break;   // EAGAIN
        NVN_BLOCKING();
        plat_sleep_ms(1);
    }
    ret_x(c, (uint64_t)r);
}

// ssize_t Write(int fd, const void *buf, size_t n)
static void ov_eventfd_write(Cpu *c) {
    int fd = (int)arg_x(c, 0);
    const uint64_t *buf = arg_p(c, 1);
    uint64_t n = arg_x(c, 2);
    int64_t r = -1;
    plat_mutex_lock(&lock);
    EventFd *e = find(fd);
    if (e && buf && n >= 8) {
        e->counter += *buf;
        r = 8;
    }
    plat_mutex_unlock(&lock);
    ret_x(c, (uint64_t)r);
}

// int Close(int fd)
static void ov_eventfd_close(Cpu *c) {
    int fd = (int)arg_x(c, 0);
    int r = -1;
    plat_mutex_lock(&lock);
    EventFd *e = find(fd);
    if (e) *e = (EventFd){0}, r = 0;
    plat_mutex_unlock(&lock);
    ret_x(c, (uint64_t)(int64_t)r);
}

// int Poll(pollfd *fds, nfds_t n, int timeout_ms): {int fd; short events; short revents}. Returns the number of descriptors
// with an event, waiting up to the timeout (negative: forever) for the first one.
static void ov_eventfd_poll(Cpu *c) {
    struct pfd { int32_t fd; int16_t events, revents; } *fds = arg_p(c, 0);
    uint64_t n = arg_x(c, 1);
    int32_t timeout = (int32_t)arg_x(c, 2);
    if (!fds && n) { ret_x(c, ~0ull); return; }
    uint64_t waited_ms = 0;
    int ready;
    for (;;) {
        ready = 0;
        plat_mutex_lock(&lock);
        for (uint64_t i = 0; i < n; i++) {
            EventFd *e = find(fds[i].fd);
            int16_t r = 0;
            if (fds[i].fd < 0) r = 0;
            else if (!e) r = P_NVAL;
            else r = (int16_t)(((fds[i].events & P_IN) && e->counter ? P_IN : 0) | ((fds[i].events & P_OUT) ? P_OUT : 0));
            fds[i].revents = r;
            ready += r != 0;
        }
        plat_mutex_unlock(&lock);
        if (ready || timeout == 0 || (timeout > 0 && waited_ms >= (uint64_t)timeout)) break;
        {
            NVN_BLOCKING();
            plat_sleep_ms(1);
        }
        waited_ms++;
    }
    ret_x(c, (uint64_t)(int64_t)ready);
}

// int Fcntl(int fd, int cmd, ...) and Ioctl(...): the descriptor's flags are fixed at creation, so setting them is accepted
// and asking for them reads 0
static void ov_eventfd_ok(Cpu *c) { ret_x(c, 0); }

HOST_OVERRIDE("_ZN2nn6socket7EventFdEmNS0_12EventFdFlagsE", ov_eventfd);
HOST_OVERRIDE("_ZN2nn6socket4ReadEiPvm", ov_eventfd_read);
HOST_OVERRIDE("_ZN2nn6socket5WriteEiPKvm", ov_eventfd_write);
HOST_OVERRIDE("_ZN2nn6socket5CloseEi", ov_eventfd_close);
HOST_OVERRIDE("_ZN2nn6socket4PollEP6pollfdmi", ov_eventfd_poll);
HOST_OVERRIDE("_ZN2nn6socket4PollEPNS0_6PollFdEmi", ov_eventfd_poll);
HOST_OVERRIDE("_ZN2nn6socket5FcntlEiiz", ov_eventfd_ok);
HOST_OVERRIDE("_ZN2nn6socket5FcntlEiNS0_12FcntlCommandEz", ov_eventfd_ok);
HOST_OVERRIDE("_ZN2nn6socket5IoctlEijPvm", ov_eventfd_ok);
HOST_OVERRIDE("_ZN2nn6socket5IoctlEiNS0_12IoctlCommandEPvm", ov_eventfd_ok);
HOST_OVERRIDE("_ZN2nn6socket5IoctlEiNS_4util7BitPackIjNS_4anif17IoctlRequestFieldEEEPvm", ov_eventfd_ok);
