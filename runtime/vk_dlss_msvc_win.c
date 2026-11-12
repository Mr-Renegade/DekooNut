// What NVIDIA's Windows NGX loader (nvsdk_ngx_s.lib, built with Visual C++; vk_dlss.c calls it) expects of the Visual
// C++ runtime and the llvm-mingw toolchain does not have: twelve symbols. Everything else the library needs is the
// Windows API and the universal C runtime, which both toolchains share. tools/build_game.py compiles this file only
// when the DLSS SDK is there, and tells the linker to leave out the Visual C++ libraries the loader names.
#include <windows.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>

// ---- /GS stack cookies
uintptr_t __security_cookie = 0x00002B992DDFA232ull;
void __security_check_cookie(uintptr_t cookie) {
    if (cookie != __security_cookie) __fastfail(2 /* FAST_FAIL_STACK_COOKIE_CHECK_FAILURE */);
}
// The exception handlers of functions with a cookie. Without C++ unwinding data: nothing to do. With it: the C++
// handler, whose data starts at the same place.
EXCEPTION_DISPOSITION __CxxFrameHandler3(PEXCEPTION_RECORD, void *, PCONTEXT, PDISPATCHER_CONTEXT);
EXCEPTION_DISPOSITION __GSHandlerCheck(PEXCEPTION_RECORD rec, void *frame, PCONTEXT ctx, PDISPATCHER_CONTEXT disp) {
    (void)rec, (void)frame, (void)ctx, (void)disp;
    return ExceptionContinueSearch;
}
EXCEPTION_DISPOSITION __GSHandlerCheck_EH(PEXCEPTION_RECORD rec, void *frame, PCONTEXT ctx, PDISPATCHER_CONTEXT disp) {
    return __CxxFrameHandler3(rec, frame, ctx, disp);
}

// ---- floating point is used
int _fltused = 0x9875;

// ---- thread-safe static locals ("magic statics"), as vcruntime's thread_safe_statics.cpp does them
_Thread_local int _Init_thread_epoch = INT_MIN;   // the library reads it through the thread's TLS block
static int init_global_epoch = INT_MIN;
static SRWLOCK init_lock = SRWLOCK_INIT;
static CONDITION_VARIABLE init_done = CONDITION_VARIABLE_INIT;
void _Init_thread_header(int *once) {
    AcquireSRWLockExclusive(&init_lock);
    if (*once == 0) {
        *once = -1;   // this thread initialises it
    } else {
        while (*once == -1) {   // another thread is at it
            SleepConditionVariableSRW(&init_done, &init_lock, 100, 0);
            if (*once == 0) {   // that thread gave up (an exception)
                *once = -1;
                ReleaseSRWLockExclusive(&init_lock);
                return;
            }
        }
        _Init_thread_epoch = init_global_epoch;
    }
    ReleaseSRWLockExclusive(&init_lock);
}
void _Init_thread_footer(int *once) {
    AcquireSRWLockExclusive(&init_lock);
    *once = ++init_global_epoch;
    _Init_thread_epoch = init_global_epoch;
    ReleaseSRWLockExclusive(&init_lock);
    WakeAllConditionVariable(&init_done);
}

// ---- C++: operator new and delete, and the standard library's two "throw" helpers, under their Visual C++ names
void *ngx_new(size_t n) __asm__("??2@YAPEAX_K@Z");
void *ngx_new(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) abort();
    return p;
}
void ngx_delete(void *p) __asm__("??3@YAXPEAX@Z");
void ngx_delete(void *p) { free(p); }
void ngx_bad_alloc(void) __asm__("?_Xbad_alloc@std@@YAXXZ");
void ngx_bad_alloc(void) { abort(); }
void ngx_length_error(const char *what) __asm__("?_Xlength_error@std@@YAXPEBD@Z");
void ngx_length_error(const char *what) { (void)what; abort(); }
