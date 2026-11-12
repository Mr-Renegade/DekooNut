// SPDX-License-Identifier: GPL-3.0-or-later
// "DekooNut Setup.exe": setup as one file for Windows (tools/make_setup_exe.py). The files of the checkout are a zip
// at the end of this program. It unpacks them into a folder DekooNut\src beside itself (DEKOONUT_HOME names another
// folder) and starts setup.bat there with the arguments it was given. Nothing of the game is in it.
#include <windows.h>
#include <stdio.h>
#include <wchar.h>

#define CAP 2048

static void fail(const wchar_t *what, const wchar_t *path) {
    wchar_t text[CAP * 2];
    swprintf(text, CAP * 2, L"%ls\n\n%ls", what, path);
    MessageBoxW(NULL, text, L"DekooNut Setup", MB_ICONERROR);
}

static int exists(const wchar_t *path) { return GetFileAttributesW(path) != INVALID_FILE_ATTRIBUTES; }

// The first start unpacks the files and fetches Python before the setup window can open. No console is shown for
// that (a console window reads as something gone wrong, and closing it ends setup): this small notice stands in
// until the setup window is there. Later starts show only the setup window.
static HWND g_notice;

static void notice_open(void) {
    int w = 440, h = 130;
    // "#32770" is the dialog class: a plain captioned window with the dialog background, no class of our own.
    g_notice = CreateWindowExW(WS_EX_TOPMOST, L"#32770", L"DekooNut Setup", WS_OVERLAPPED | WS_CAPTION,
                               (GetSystemMetrics(SM_CXSCREEN) - w) / 2, (GetSystemMetrics(SM_CYSCREEN) - h) / 2, w, h,
                               NULL, NULL, GetModuleHandleW(NULL), NULL);
    if (!g_notice) return;
    RECT r;
    GetClientRect(g_notice, &r);
    HWND text = CreateWindowExW(0, L"STATIC",
                                L"\nGetting ready: fetching Python (once).\nThe setup window opens in a moment ...",
                                WS_CHILD | WS_VISIBLE | SS_CENTER, 0, 0, r.right, r.bottom, g_notice, NULL,
                                GetModuleHandleW(NULL), NULL);
    NONCLIENTMETRICSW m = {sizeof m};
    if (text && SystemParametersInfoW(SPI_GETNONCLIENTMETRICS, sizeof m, &m, 0))
        SendMessageW(text, WM_SETFONT, (WPARAM)CreateFontIndirectW(&m.lfMessageFont), TRUE);
    ShowWindow(g_notice, SW_SHOWNORMAL);
    UpdateWindow(g_notice);
}

static void notice_close(void) {
    if (g_notice) DestroyWindow(g_notice);
    g_notice = NULL;
}

// Run a command line without a console window and wait; its exit code, or -1 when it could not start. While it
// runs the notice stays alive, and goes once the setup window (SDL's class, setup/gui/main.cpp's title) is there.
static int run(wchar_t *cmd, const wchar_t *cwd) {
    STARTUPINFOW si = {sizeof si};
    PROCESS_INFORMATION pi;
    if (!CreateProcessW(NULL, cmd, NULL, NULL, FALSE, CREATE_NO_WINDOW, NULL, cwd, &si, &pi)) return -1;
    while (MsgWaitForMultipleObjects(1, &pi.hProcess, FALSE, 200, QS_ALLINPUT) != WAIT_OBJECT_0) {
        MSG msg;
        while (PeekMessageW(&msg, NULL, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        if (g_notice && FindWindowW(L"SDL_app", L"DekooNut Setup")) notice_close();
    }
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return (int)code;
}

int WINAPI wWinMain(HINSTANCE inst, HINSTANCE prev, PWSTR args, int show) {
    (void)inst, (void)prev, (void)show;
    static wchar_t exe[CAP], home[CAP], src[CAP], id[CAP], probe[CAP], cmd[CAP * 4], sys[CAP];
    if (!GetModuleFileNameW(NULL, exe, CAP)) return 1;
    if (!GetEnvironmentVariableW(L"DEKOONUT_HOME", home, CAP)) {
        wcscpy(home, exe);
        wchar_t *cut = wcsrchr(home, L'\\');
        if (!cut) return 1;
        wcscpy(cut + 1, L"DekooNut");
    }
    swprintf(src, CAP, L"%ls\\src", home);
    swprintf(id, CAP, L"%ls\\.release-id", src);
    swprintf(probe, CAP, L"%ls\\toolchain\\python\\tools\\python.exe", src);
    int first = !exists(probe);   // setup.bat fetches Python then
    if (first) notice_open();

    // This program's size and time say whether the folder holds its files already.
    WIN32_FILE_ATTRIBUTE_DATA fa;
    char want[96] = "", have[96] = "";
    if (!GetFileAttributesExW(exe, GetFileExInfoStandard, &fa)) return 1;
    snprintf(want, sizeof want, "%lu-%lu-%lu", fa.nFileSizeLow, fa.ftLastWriteTime.dwHighDateTime, fa.ftLastWriteTime.dwLowDateTime);
    FILE *f = _wfopen(id, L"rb");
    if (f) {
        size_t n = fread(have, 1, sizeof have - 1, f);
        have[n] = 0;
        fclose(f);
    }
    if (strcmp(want, have)) {
        CreateDirectoryW(home, NULL);
        CreateDirectoryW(src, NULL);
        if (!exists(src)) { notice_close(); fail(L"This folder could not be made. Put the program somewhere you may write, or set DEKOONUT_HOME.", src); return 1; }
        GetSystemDirectoryW(sys, CAP);
        // -m: files get the time of unpacking, so the window's program (last in the zip) is newer than its sources.
        swprintf(cmd, CAP * 4, L"\"%ls\\tar.exe\" -xmf \"%ls\" -C \"%ls\"", sys, exe, src);
        if (run(cmd, NULL)) { notice_close(); fail(L"The files could not be unpacked (tar.exe of Windows 10 or newer is needed).", src); return 1; }
        f = _wfopen(id, L"wb");
        if (f) fputs(want, f), fclose(f);
    }
    swprintf(cmd, CAP * 4, L"cmd.exe /c \"\"%ls\\setup.bat\" %ls\"", src, args ? args : L"");
    int code = run(cmd, src);
    notice_close();
    if (code == -1) fail(L"Setup could not be started.", src);
    else if (first && !exists(probe))   // what setup.bat says about it went to no console
        fail(L"Python could not be fetched. Check the internet connection, then start DekooNut Setup again.", src);
    return code;
}
