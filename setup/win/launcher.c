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

// Run a command line and wait; its exit code, or -1 when it could not start.
static int run(wchar_t *cmd, const wchar_t *cwd, DWORD flags) {
    STARTUPINFOW si = {sizeof si};
    PROCESS_INFORMATION pi;
    if (!CreateProcessW(NULL, cmd, NULL, NULL, FALSE, flags, NULL, cwd, &si, &pi)) return -1;
    WaitForSingleObject(pi.hProcess, INFINITE);
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
        if (!exists(src)) { fail(L"This folder could not be made. Put the program somewhere you may write, or set DEKOONUT_HOME.", src); return 1; }
        GetSystemDirectoryW(sys, CAP);
        // -m: files get the time of unpacking, so the window's program (last in the zip) is newer than its sources.
        swprintf(cmd, CAP * 4, L"\"%ls\\tar.exe\" -xmf \"%ls\" -C \"%ls\"", sys, exe, src);
        if (run(cmd, NULL, CREATE_NO_WINDOW)) { fail(L"The files could not be unpacked (tar.exe of Windows 10 or newer is needed).", src); return 1; }
        f = _wfopen(id, L"wb");
        if (f) fputs(want, f), fclose(f);
    }
    // The first start fetches Python: a console shows that. Later starts show only the window.
    swprintf(probe, CAP, L"%ls\\toolchain\\python\\tools\\python.exe", src);
    DWORD flags = exists(probe) ? CREATE_NO_WINDOW : CREATE_NEW_CONSOLE;
    swprintf(cmd, CAP * 4, L"cmd.exe /c \"\"%ls\\setup.bat\" %ls\"", src, args ? args : L"");
    int code = run(cmd, src, flags);
    if (code == -1) fail(L"Setup could not be started.", src);
    return code;
}
