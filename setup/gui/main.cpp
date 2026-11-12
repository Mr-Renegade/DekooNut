// SPDX-License-Identifier: GPL-3.0-or-later
// The setup window: a front end over `python3 -m setup` (setup/cli.py). It collects the game file, the install folder
// and the quality level, starts the backend with --json and shows its events (setup/events.py). It decides nothing
// about the build itself. The scene is drawn here: the top of a stump with the nut at its heart, whose rings are the
// build's steps.
//
//   dekoonut-setup --root DIR --python EXE [--backend "CMD ..."] [--size WxH] [--shot FILE --shot-at S] [--autobuild] [--click X,Y]
//
// --backend replaces `EXE -m setup` (tests/setup_fake.py plays a build); --shot saves the window as a PNG after S
// seconds and exits; --autobuild presses Build once the form is filled; --click X,Y clicks there after a second.
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <utility>
#include <vector>

#include "imgui.h"
#include "imgui/imgui_impl_sdl3.h"
#include "imgui/imgui_impl_sdlrenderer3.h"

// ---- JSON, as much as the backend's events need

struct J {
    enum T { NUL, BOOL, NUM, STR, ARR, OBJ } t = NUL;
    bool b = false;
    double n = 0;
    std::string s;
    std::vector<J> a;
    std::vector<std::pair<std::string, J>> o;
    const J &operator[](const char *k) const {
        static const J none;
        for (auto &kv : o)
            if (kv.first == k) return kv.second;
        return none;
    }
};

static void jspace(const char *&p) { while (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r') p++; }

static bool jstring(const char *&p, std::string &out) {
    if (*p != '"') return false;
    for (p++; *p && *p != '"'; p++) {
        if (*p != '\\') { out += *p; continue; }
        switch (*++p) {
        case 'n': out += '\n'; break;
        case 't': out += '\t'; break;
        case 'r': break;
        case 'b': case 'f': break;
        case 'u': {
            unsigned c = 0;
            for (int i = 0; i < 4 && p[1]; i++) c = c * 16 + (unsigned)strtol(std::string(1, *++p).c_str(), NULL, 16);
            if (c < 0x80) out += (char)c;
            else if (c < 0x800) out += (char)(0xC0 | c >> 6), out += (char)(0x80 | (c & 0x3F));
            else out += (char)(0xE0 | c >> 12), out += (char)(0x80 | ((c >> 6) & 0x3F)), out += (char)(0x80 | (c & 0x3F));
            break;
        }
        case 0: return false;
        default: out += *p;
        }
    }
    if (*p != '"') return false;
    p++;
    return true;
}

static bool jparse(const char *&p, J &v) {
    jspace(p);
    if (*p == '"') { v.t = J::STR; return jstring(p, v.s); }
    if (*p == '{') {
        v.t = J::OBJ;
        for (p++;;) {
            jspace(p);
            if (*p == '}') { p++; return true; }
            std::string k;
            if (!jstring(p, k)) return false;
            jspace(p);
            if (*p++ != ':') return false;
            J child;
            if (!jparse(p, child)) return false;
            v.o.emplace_back(std::move(k), std::move(child));
            jspace(p);
            if (*p == ',') p++;
        }
    }
    if (*p == '[') {
        v.t = J::ARR;
        for (p++;;) {
            jspace(p);
            if (*p == ']') { p++; return true; }
            J child;
            if (!jparse(p, child)) return false;
            v.a.push_back(std::move(child));
            jspace(p);
            if (*p == ',') p++;
        }
    }
    if (!strncmp(p, "true", 4)) { v.t = J::BOOL, v.b = true, p += 4; return true; }
    if (!strncmp(p, "false", 5)) { v.t = J::BOOL, p += 5; return true; }
    if (!strncmp(p, "null", 4)) { p += 4; return true; }
    char *end;
    v.n = strtod(p, &end);
    if (end == p) return false;
    v.t = J::NUM, p = end;
    return true;
}

// ---- the backend process

struct Proc {
    SDL_Process *p = nullptr;
    std::string buf;

    bool start(const std::vector<std::string> &args) {
        std::vector<const char *> argv;
        for (auto &a : args) argv.push_back(a.c_str());
        argv.push_back(nullptr);
        SDL_PropertiesID props = SDL_CreateProperties();
        SDL_SetPointerProperty(props, SDL_PROP_PROCESS_CREATE_ARGS_POINTER, (void *)argv.data());
        SDL_SetNumberProperty(props, SDL_PROP_PROCESS_CREATE_STDOUT_NUMBER, SDL_PROCESS_STDIO_APP);
        SDL_SetNumberProperty(props, SDL_PROP_PROCESS_CREATE_STDIN_NUMBER, SDL_PROCESS_STDIO_NULL);
#ifdef _WIN32
        // The window has no console, so Windows would open a console window for the backend (a console program) and
        // keep it open for the whole build. "Background" is CREATE_NO_WINDOW there: a hidden console, which the tools
        // the backend starts inherit. SDL then reports exit code 0 for it; the events say how a build ended.
        SDL_SetBooleanProperty(props, SDL_PROP_PROCESS_CREATE_BACKGROUND_BOOLEAN, true);
#endif
        p = SDL_CreateProcessWithProperties(props);
        SDL_DestroyProperties(props);
        buf.clear();
        return p != nullptr;
    }
    // The lines that arrived; false once the process has ended and everything is read.
    bool poll(std::vector<std::string> &lines, int *code) {
        if (!p) return false;
        bool ended = SDL_WaitProcess(p, false, code);
        char tmp[8192];
        size_t n;
        SDL_IOStream *out = SDL_GetProcessOutput(p);
        while (out && (n = SDL_ReadIO(out, tmp, sizeof tmp)) > 0) buf.append(tmp, n);
        size_t at;
        while ((at = buf.find('\n')) != std::string::npos) {
            lines.push_back(buf.substr(0, at));
            buf.erase(0, at + 1);
        }
        if (!ended) return true;
        SDL_DestroyProcess(p);
        p = nullptr;
        return false;
    }
    void cancel() { if (p) SDL_KillProcess(p, false); }
};

// ---- state

struct Level { std::string id, name, about; };
struct StepRow { std::string id, title; int state = 0; };   // 0 waiting, 1 running, 2 done
enum Card { CARD_NONE, CARD_BUILD, CARD_ERROR, CARD_DONE };   // what the column shows: the form, or one of these

static struct App {
    std::string root, python, shot, click;
    std::vector<std::string> backend;
    float shot_at = 2;
    bool autobuild = false;

    // from --info
    bool info_ready = false, installed = false, can_install = false;
    bool desktop = false, desktop_known = false;   // the tick box "a shortcut on the desktop" (Windows)
    std::string version, game, dest, gpu, install_command;
    std::vector<std::string> missing;
    std::vector<Level> levels;
    int level = 0;
    std::string suggested, suggested_name, suggested_why;

    // a build
    Card card = CARD_NONE;
    bool card_opened = false, show_log = false, cancelling = false;
    std::vector<StepRow> steps;
    int cur = -1;
    float frac = 0, shown = 0, ring_frac = 0;
    std::string progress, err_text, err_detail, err_command, log_path, play;
    std::vector<std::string> log, warnings;
    Uint64 started = 0, ended = 0;

    // the scene
    SDL_Texture *nut = nullptr;
    float flash = 0, energy = 0;
    std::string copied;
    Uint64 copied_at = 0;
} A;

static Proc g_info, g_build;
static SDL_Window *g_win;
static SDL_Renderer *g_ren;
static SDL_Mutex *g_pick_lock;
static std::string g_picked_game, g_picked_dest;   // set by the dialog callbacks, on any thread
static ImFont *F_TITLE, *F_HEAD, *F_ROW, *F_BODY, *F_SMALL, *F_MONO;
static float g_font_scale;

#ifdef _WIN32
#define LAUNCHER "\\dekoonut.bat"
#define MENU "Start menu"
#else
#define LAUNCHER "/dekoonut"
#define MENU "applications menu"
#endif

// By step (setup/steps.py all_steps): ..., the game, translating the shaders, compiling them, the level, finishing.
static const float STEP_WEIGHT[] = {1, 3, 1, 1, 12, 10, 8, 6, 30, 12, 25, 1, 1};

static ImU32 C(float r, float g, float b, float a = 1) { return ImGui::ColorConvertFloat4ToU32(ImVec4(r, g, b, a)); }
static const ImVec4 GOLD(0.96f, 0.74f, 0.28f, 1), TEXT(0.96f, 0.92f, 0.84f, 1), DIM(0.66f, 0.58f, 0.47f, 1);

static float rnd(int i, int k) {
    unsigned h = (unsigned)i * 374761393u + (unsigned)k * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return ((h ^ (h >> 16)) & 0xFFFFFF) / (float)0x1000000;
}

static bool is_dir(const std::string &p) {
    SDL_PathInfo info;
    return SDL_GetPathInfo(p.c_str(), &info) && info.type == SDL_PATHTYPE_DIRECTORY;
}

static std::string base_name(const std::string &p) {
    size_t at = p.find_last_of("/\\");
    return at == std::string::npos ? p : p.substr(at + 1);
}

// ---- backend

static std::vector<std::string> backend_args(std::initializer_list<std::string> more) {
    std::vector<std::string> v = A.backend;
    v.insert(v.end(), more);
    return v;
}

static void ask_info() {
    A.info_ready = false;
    char size[32] = "";
    if (const SDL_DisplayMode *mode = SDL_GetDesktopDisplayMode(SDL_GetPrimaryDisplay())) snprintf(size, sizeof size, "%dx%d", mode->w, mode->h);
    g_info.start(size[0] ? backend_args({"--info", "--display", size}) : backend_args({"--info"}));
}

static void take_info(const std::string &line) {
    J j;
    const char *p = line.c_str();
    if (!jparse(p, j) || j.t != J::OBJ) return;
    A.version = j["version"].s;
    if (A.game.empty()) A.game = j["game"].s;
    if (A.dest.empty()) A.dest = j["dest"].s;
    A.gpu = j["gpu"].s;
    if (!A.desktop_known) A.desktop = j["desktop"].b, A.desktop_known = true;   // later answers leave the tick as it is
    A.installed = j["installed"].b;
    A.install_command = j["install_command"].s;
    A.can_install = j["can_install"].b;
    A.missing.clear();
    for (auto &m : j["missing"].a) A.missing.push_back(m.s);
    A.levels.clear();
    for (auto &l : j["levels"].a) A.levels.push_back({l["id"].s, l["name"].s, l["about"].s});
    A.suggested = j["suggested"].s, A.suggested_why = j["suggested_why"].s;
    for (size_t i = 0; i < A.levels.size(); i++) {
        if (A.levels[i].id == j["quality"].s) A.level = (int)i;
        if (A.levels[i].id == A.suggested) A.suggested_name = A.levels[i].name;
    }
    A.info_ready = true;
}

static void load_nut(const std::string &path) {
    SDL_Surface *s = SDL_LoadPNG(path.c_str());
    if (!s) return;
    SDL_Texture *t = SDL_CreateTextureFromSurface(g_ren, s);
    SDL_DestroySurface(s);
    if (!t) return;
    SDL_SetTextureScaleMode(t, SDL_SCALEMODE_LINEAR);
    if (A.nut) SDL_DestroyTexture(A.nut);
    A.nut = t;
}

static void start_run(const std::vector<std::string> &args);

static void start_build() {
    if (!A.missing.empty()) {   // the backend would say the same after a moment
        A.err_text = "This PC needs a few build tools first.";
        A.err_detail = "Missing: ";
        for (size_t i = 0; i < A.missing.size(); i++) A.err_detail += (i ? ", " : "") + A.missing[i];
        A.err_detail += A.install_command.empty() ? "\nInstall them with the system's package manager, then check again."
                      : A.can_install ? "\nSetup can install them (it asks for your password), or run this in a terminal:"
                                      : "\nRun this in a terminal, then check again:";
        A.err_command = A.install_command;
        A.card = CARD_ERROR, A.card_opened = false;
        return;
    }
    std::vector<std::string> args = backend_args({"--json", "--game", A.game, "--dest", A.dest});
    if (!A.levels.empty()) args.push_back("--quality"), args.push_back(A.levels[A.level].id);
    if (A.desktop) args.push_back("--desktop-entry");
    start_run(args);
}

// Start the backend and show its steps: a build, or the installation of the system packages.
static void start_run(const std::vector<std::string> &args) {
    A.steps.clear(), A.log.clear(), A.warnings.clear();
    A.cur = -1, A.frac = A.shown = 0, A.progress.clear();
    A.err_text.clear(), A.err_detail.clear(), A.err_command.clear(), A.play.clear();
    A.cancelling = false, A.show_log = false;
    A.started = SDL_GetTicks(), A.ended = 0;
    if (!g_build.start(args)) {
        A.err_text = "Setup could not be started.", A.err_detail = SDL_GetError();
        A.card = CARD_ERROR, A.card_opened = false;
        return;
    }
    A.card = CARD_BUILD, A.card_opened = false;
}

static int step_index(const std::string &id) {
    for (size_t i = 0; i < A.steps.size(); i++)
        if (A.steps[i].id == id) return (int)i;
    return -1;
}

static void take_event(const std::string &line) {
    J j;
    const char *p = line.c_str();
    if (!jparse(p, j) || j.t != J::OBJ) {
        A.log.push_back(line);
        return;
    }
    const std::string &type = j["type"].s;
    if (type == "start") {
        A.log_path = j["log"].s;
        for (auto &s : j["steps"].a) A.steps.push_back({s["id"].s, s["title"].s, 0});
    } else if (type == "step") {
        int i = step_index(j["step"].s);
        if (i < 0) return;
        if (j["state"].s == "started") A.steps[i].state = 1, A.cur = i, A.frac = 0, A.progress.clear();
        else A.steps[i].state = 2, A.frac = 0;
    } else if (type == "progress") {
        A.progress = j["text"].s;
        if (j["total"].n > 0) A.frac = (float)(j["done"].n / j["total"].n);
    } else if (type == "log") {
        A.log.push_back(j["line"].s);
        if (A.log.size() > 600) A.log.erase(A.log.begin(), A.log.begin() + 200);
    } else if (type == "icon") {
        load_nut(j["path"].s);
        A.flash = 1;
    } else if (type == "warning") {
        A.warnings.push_back(j["text"].s);
        A.log.push_back("warning: " + j["text"].s);
    } else if (type == "error") {
        A.err_text = j["text"].s, A.err_detail = j["detail"].s, A.err_command = j["command"].s;
    } else if (type == "end") {
        A.ended = SDL_GetTicks();
        A.play = j["play"].s;
        if (j["result"].s == "ok" && A.play.empty()) A.card = CARD_NONE, ask_info();   // the packages are in: back to the form
        else if (j["result"].s == "ok") A.card = CARD_DONE, A.flash = 1, A.installed = true;
        else if (j["result"].s == "cancelled") A.card = CARD_NONE;
        else A.card = CARD_ERROR;
        A.card_opened = false;
    }
}

static float overall() {
    float done = 0, total = 0;
    for (size_t i = 0; i < A.steps.size(); i++) {
        float w = i < SDL_arraysize(STEP_WEIGHT) ? STEP_WEIGHT[i] : 1;
        total += w;
        if (A.steps[i].state == 2) done += w;
        else if (A.steps[i].state == 1) done += w * A.frac;
    }
    return total > 0 ? done / total : 0;
}

static void poll_backend() {
    std::vector<std::string> lines;
    int code = 0;
    if (g_info.p) {
        g_info.poll(lines, &code);
        for (auto &l : lines) take_info(l);
        lines.clear();
    }
    if (g_build.p) {
        bool alive = g_build.poll(lines, &code);
        for (auto &l : lines) take_event(l);
        if (!alive && A.card == CARD_BUILD) {   // it ended without saying so
            if (A.cancelling) A.card = CARD_NONE;
            else {
                if (A.err_text.empty()) A.err_text = "Setup stopped unexpectedly.";
                A.card = CARD_ERROR, A.card_opened = false;
            }
        }
    }
}

// ---- file dialogs

static void SDLCALL picked_game(void *, const char *const *files, int) {
    if (!files || !files[0]) return;
    SDL_LockMutex(g_pick_lock);
    g_picked_game = files[0];
    SDL_UnlockMutex(g_pick_lock);
}

static void SDLCALL picked_dest(void *, const char *const *files, int) {
    if (!files || !files[0]) return;
    SDL_LockMutex(g_pick_lock);
    g_picked_dest = files[0];
    SDL_UnlockMutex(g_pick_lock);
}

static void take_picks() {
    SDL_LockMutex(g_pick_lock);
    if (!g_picked_game.empty()) A.game = g_picked_game, g_picked_game.clear();
    if (!g_picked_dest.empty()) A.dest = g_picked_dest, g_picked_dest.clear();
    SDL_UnlockMutex(g_pick_lock);
}

// ---- the scene: the top of a great stump, the nut at the heart of its rings. The rings around the nut are the
// build's steps, one each: a finished step's ring glows, the running one fills clockwise.

static const float TAU = 6.2831853f;

// A slightly uneven ring (growth rings are never circles), from angle a0 to a1, clockwise from the top.
static void ring(ImDrawList *d, ImVec2 c, float r, int k, float a0, float a1, ImU32 col, float thick) {
    int n = (int)(fabsf(a1 - a0) / TAU * 140) + 2;
    d->PathClear();
    for (int i = 0; i <= n; i++) {
        float a = a0 + (a1 - a0) * i / n;
        float rr = r * (1 + 0.011f * sinf(3 * a + k * 1.7f) + 0.006f * sinf(7 * a + k * 0.9f));
        d->PathLineTo(ImVec2(c.x + sinf(a) * rr, c.y - cosf(a) * rr));
    }
    d->PathStroke(col, 0, thick);
}

static void scene(ImDrawList *d, float W, float H, float s, float t) {
    float e = A.energy;
    ImVec2 c(W * 0.305f, H * 0.52f);
    d->AddRectFilled(ImVec2(0, 0), ImVec2(W, H), C(0.090f, 0.060f, 0.040f));
    for (int i = 0; i < 26; i++)   // heartwood: warmer towards the middle
        d->AddCircleFilled(c, (1250 - 46 * i) * s, C(0.42f, 0.25f, 0.11f, 0.016f + 0.004f * e), 72);
    float far = sqrtf(W * W + H * H);
    for (int k = 0; k * 21 * s < far; k++) {   // the grain
        float r = (40 + k * 21) * s + rnd(k, 1) * 9 * s, v = rnd(k, 2);
        ring(d, c, r, k, 0, TAU, v > 0.5f ? C(0.20f, 0.13f, 0.075f, 0.55f) : C(0.045f, 0.028f, 0.018f, 0.50f), (1.5f + 3.5f * rnd(k, 3)) * s);
    }
    for (int i = 0; i < 6; i++) {   // drying cracks run outwards
        float a = TAU * (0.07f + i / 6.f) + rnd(i, 4), r0 = (560 + 200 * rnd(i, 5)) * s, w = (5 + 6 * rnd(i, 6)) * s;
        ImVec2 dir(sinf(a), -cosf(a)), side(-dir.y, dir.x);
        d->AddTriangleFilled(ImVec2(c.x + dir.x * r0, c.y + dir.y * r0), ImVec2(c.x + dir.x * far + side.x * w, c.y + dir.y * far + side.y * w),
                             ImVec2(c.x + dir.x * far - side.x * w, c.y + dir.y * far - side.y * w), C(0.030f, 0.018f, 0.012f, 0.75f));
    }
    // the step rings
    float r0 = 212 * s, gap = 25 * s;
    int count = A.steps.empty() ? 12 : (int)A.steps.size();
    for (int k = 0; k < count; k++) {
        float r = r0 + k * gap;
        int st = A.steps.empty() ? (A.installed ? 2 : 0) : A.steps[k].state;
        ring(d, c, r, k + 100, 0, TAU, C(0.30f, 0.20f, 0.11f, 0.85f), 3 * s);
        if (st == 2) {
            ring(d, c, r, k + 100, 0, TAU, C(1.0f, 0.72f, 0.22f, 0.16f), 11 * s);
            ring(d, c, r, k + 100, 0, TAU, C(0.96f, 0.74f, 0.28f), 4 * s);
        } else if (st == 1) {
            float f = A.ring_frac, glow = 0.5f + 0.5f * sinf(t * 4);
            ring(d, c, r, k + 100, 0, TAU, C(1.0f, 0.80f, 0.35f, 0.10f + 0.10f * glow), 9 * s);   // the running step breathes
            if (f > 0.003f) {
                ring(d, c, r, k + 100, 0, TAU * f, C(1.0f, 0.72f, 0.22f, 0.25f), 12 * s);
                ring(d, c, r, k + 100, 0, TAU * f, C(1.0f, 0.86f, 0.45f), 5 * s);
                float a = TAU * f, rr = r * (1 + 0.011f * sinf(3 * a + (k + 100) * 1.7f) + 0.006f * sinf(7 * a + (k + 100) * 0.9f));
                d->AddCircleFilled(ImVec2(c.x + sinf(a) * rr, c.y - cosf(a) * rr), 7 * s, C(1, 0.95f, 0.75f), 16);
            }
        }
    }
    // the nut and its light
    float size = 330 * s, bob = sinf(t * 1.5f) * 6 * s, pulse = 1 + 0.05f * sinf(t * 2.1f);
    ImVec2 n(c.x, c.y - 8 * s + bob);
    for (int i = 0; i < 24; i++)
        d->AddCircleFilled(c, (60 + 13 * i) * s * pulse, C(1.0f, 0.66f, 0.18f, (0.016f + 0.030f * e) * (1 - i / 24.f)), 56);
    d->AddEllipseFilled(ImVec2(c.x + 6 * s, c.y + size * 0.36f), ImVec2(size * 0.36f - bob, size * 0.085f), C(0, 0, 0, 0.42f), 0, 40);
    if (A.nut) {
        d->AddImage((ImTextureID)(intptr_t)A.nut, ImVec2(n.x - size / 2, n.y - size / 2), ImVec2(n.x + size / 2, n.y + size / 2));
    } else {
        d->AddEllipseFilled(n, ImVec2(size * 0.42f, size * 0.33f), C(0.45f, 0.29f, 0.16f), -0.3f, 48);
        d->AddEllipseFilled(n, ImVec2(size * 0.30f, size * 0.06f), C(1.0f, 0.78f, 0.25f), -0.3f, 32);
    }
    for (int i = 0; i < 44; i++) {   // sparks rise from the nut's light
        float sp = 0.035f + 0.05f * rnd(i, 7), ph = fmodf(t * sp + rnd(i, 8), 1.0f), a = sinf(ph * 3.14159f) * (0.35f + 0.65f * e);
        float x = c.x + (rnd(i, 9) - 0.5f) * 1100 * s + sinf(t * 0.6f + i) * 26 * s, y = c.y + 420 * s - ph * 940 * s;
        d->AddCircleFilled(ImVec2(x, y), 6 * s, C(1, 0.70f, 0.25f, a * 0.10f), 12);
        d->AddCircleFilled(ImVec2(x, y), 1.8f * s, C(1, 0.90f, 0.60f, a * 0.9f), 8);
    }
    // darker towards the edges, and under the text on the right
    d->AddRectFilledMultiColor(ImVec2(0, 0), ImVec2(W, H * 0.30f), C(0, 0, 0, 0.55f), C(0, 0, 0, 0.55f), C(0, 0, 0, 0), C(0, 0, 0, 0));
    d->AddRectFilledMultiColor(ImVec2(0, H * 0.70f), ImVec2(W, H), C(0, 0, 0, 0), C(0, 0, 0, 0), C(0, 0, 0, 0.60f), C(0, 0, 0, 0.60f));
    d->AddRectFilledMultiColor(ImVec2(0, 0), ImVec2(W * 0.10f, H), C(0, 0, 0, 0.5f), C(0, 0, 0, 0), C(0, 0, 0, 0), C(0, 0, 0, 0.5f));
    d->AddRectFilledMultiColor(ImVec2(W * 0.50f, 0), ImVec2(W * 0.64f, H), C(0.03f, 0.02f, 0.012f, 0), C(0.03f, 0.02f, 0.012f, 0.80f), C(0.03f, 0.02f, 0.012f, 0.80f), C(0.03f, 0.02f, 0.012f, 0));
    d->AddRectFilled(ImVec2(W * 0.64f, 0), ImVec2(W, H), C(0.03f, 0.02f, 0.012f, 0.80f));
    if (A.flash > 0.01f) {   // the nut goes off
        float f = A.flash;
        d->AddRectFilled(ImVec2(0, 0), ImVec2(W, H), C(1, 0.96f, 0.82f, f * f * 0.55f));
        d->AddCircle(n, (1 - f) * 1100 * s + size * 0.3f, C(1, 0.90f, 0.60f, f * 0.8f), 96, 10 * s * f + 1);
        for (int i = 0; i < 10; i++) d->AddCircleFilled(n, (size * 0.5f + 40 * i * s) * (1.3f - f * 0.3f), C(1, 0.92f, 0.65f, f * 0.05f), 48);
    }
}

// ---- widgets

static std::string fit_left(const std::string &text, float width) {
    if (ImGui::CalcTextSize(text.c_str()).x <= width) return text;
    for (size_t cut = 1; cut < text.size(); cut++) {
        if ((text[cut] & 0xC0) == 0x80) continue;
        std::string t = "..." + text.substr(cut);
        if (ImGui::CalcTextSize(t.c_str()).x <= width) return t;
    }
    return "...";
}

static void check_mark(ImDrawList *d, ImVec2 p, float s, ImU32 col) {
    const ImVec2 pts[3] = {ImVec2(p.x, p.y + 9 * s), ImVec2(p.x + 7 * s, p.y + 16 * s), ImVec2(p.x + 20 * s, p.y)};
    d->AddPolyline(pts, 3, col, 0, 3.5f * s);
}

static void gap(float px) { ImGui::Dummy(ImVec2(0, px)); }

static void wrapped(ImFont *font, const ImVec4 &col, const std::string &text, float w) {
    ImGui::PushFont(font);
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + w);
    ImGui::TextColored(col, "%s", text.c_str());
    ImGui::PopTextWrapPos();
    ImGui::PopFont();
}

// The heading of a numbered choice: the number, or a tick once it is made.
static void heading(int number, const char *text, bool done, float s) {
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImDrawList *d = ImGui::GetWindowDrawList();
    float r = 15 * s;
    ImVec2 c(p.x + r, p.y + r);
    if (done) {
        d->AddCircleFilled(c, r, C(0.36f, 0.56f, 0.24f), 28);
        check_mark(d, ImVec2(c.x - 8 * s, c.y - 6 * s), s * 0.8f, C(0.93f, 0.98f, 0.85f));
    } else {
        d->AddCircle(c, r, ImGui::GetColorU32(GOLD), 28, 2 * s);
        char b[4];
        snprintf(b, sizeof b, "%d", number);
        ImGui::PushFont(F_SMALL);
        ImVec2 ts = ImGui::CalcTextSize(b);
        d->AddText(ImVec2(c.x - ts.x / 2, c.y - ts.y / 2), ImGui::GetColorU32(GOLD), b);
        ImGui::PopFont();
    }
    ImGui::PushFont(F_SMALL);
    d->AddText(ImVec2(p.x + 2 * r + 12 * s, c.y - ImGui::GetFontSize() / 2), ImGui::GetColorU32(DIM), text);
    ImGui::PopFont();
    ImGui::Dummy(ImVec2(10, 2 * r + 8 * s));
}

static void push_button(bool primary, float rounding, float s) {
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, rounding);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, primary ? 0 : 1.5f * s);
    ImGui::PushStyleColor(ImGuiCol_Button, primary ? ImVec4(0.96f, 0.72f, 0.24f, 1) : ImVec4(1, 0.9f, 0.75f, 0.06f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, primary ? ImVec4(1.0f, 0.82f, 0.38f, 1) : ImVec4(1, 0.9f, 0.75f, 0.14f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, primary ? ImVec4(0.86f, 0.62f, 0.18f, 1) : ImVec4(1, 0.9f, 0.75f, 0.20f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.96f, 0.74f, 0.28f, 0.40f));
    ImGui::PushStyleColor(ImGuiCol_Text, primary ? ImVec4(0.16f, 0.09f, 0.03f, 1) : TEXT);
}

static void pop_button() {
    ImGui::PopStyleColor(5);
    ImGui::PopStyleVar(2);
}

static bool button(const char *label, float w, float s, bool primary = false, bool big = false) {
    float h = (big ? 72 : 52) * s;
    ImGui::PushFont(big ? F_HEAD : F_BODY);
    push_button(primary, h / 2, s);
    bool pressed = ImGui::Button(label, ImVec2(w, h));
    pop_button();
    ImGui::PopFont();
    return pressed;
}

// A choice shown as a field: its value (or what to choose), and "Change" at the right.
static bool field(const char *id, const std::string &value, const char *empty, float w, float s) {
    ImGui::PushFont(F_ROW);
    ImVec2 p = ImGui::GetCursorScreenPos();
    float h = 58 * s;
    bool pressed = ImGui::InvisibleButton(id, ImVec2(w, h));
    bool on = ImGui::IsItemFocused() || ImGui::IsItemHovered();
    ImDrawList *d = ImGui::GetWindowDrawList();
    d->AddRectFilled(p, ImVec2(p.x + w, p.y + h), C(1, 0.9f, 0.75f, on ? 0.12f : 0.06f), 10 * s);
    d->AddRect(p, ImVec2(p.x + w, p.y + h), C(0.96f, 0.74f, 0.28f, on ? 0.85f : 0.22f), 10 * s, 0, 1.5f * s);
    float ty = p.y + (h - ImGui::GetFontSize()) / 2;
    const char *act = value.empty() ? "Choose" : "Change";
    float aw = ImGui::CalcTextSize(act).x;
    d->AddText(ImVec2(p.x + w - aw - 20 * s, ty), ImGui::GetColorU32(on ? GOLD : DIM), act);
    std::string v = value.empty() ? empty : fit_left(value, w - aw - 64 * s);
    d->AddText(ImVec2(p.x + 20 * s, ty), ImGui::GetColorU32(value.empty() ? DIM : TEXT), v.c_str());
    ImGui::PopFont();
    return pressed;
}

static bool link(const char *label, float s) {
    ImGui::PushFont(F_SMALL);
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1, 0.9f, 0.75f, 0.10f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(1, 0.9f, 0.75f, 0.16f));
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.90f, 0.72f, 0.36f, 1));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 6 * s);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(8 * s, 5 * s));
    bool pressed = ImGui::Button(label);
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor(4);
    ImGui::PopFont();
    return pressed;
}

static std::string duration(Uint64 ms) {
    char b[48];
    unsigned sec = (unsigned)(ms / 1000);
    if (sec < 60) snprintf(b, sizeof b, "%u s", sec);
    else if (sec < 3600) snprintf(b, sizeof b, "%u min %02u s", sec / 60, sec % 60);
    else snprintf(b, sizeof b, "%u h %02u min", sec / 3600, sec % 3600 / 60);
    return b;
}

static void copy_text(const std::string &text) {
    SDL_SetClipboardText(text.c_str());
    A.copied = text, A.copied_at = SDL_GetTicks();
}

static void open_path(const std::string &path) {
    std::string url = "file://" + path;
#ifdef _WIN32
    url = "file:///" + path;
    for (char &c : url)
        if (c == '\\') c = '/';
#endif
    SDL_OpenURL(url.c_str());
}

// Stop the running build. Where ending the backend would leave its tools running, it is asked through a file it watches.
static void cancel_build() {
    A.cancelling = true;
#ifdef _WIN32
    if (SDL_IOStream *f = SDL_IOFromFile((A.dest + "\\.setup\\cancel").c_str(), "wb")) {
        SDL_CloseIO(f);
        return;
    }
#endif
    g_build.cancel();
}

static void quit() {
    SDL_Event ev = {SDL_EVENT_QUIT};
    SDL_PushEvent(&ev);
}

static void play() {
#ifdef _WIN32
    const char *argv[] = {"cmd.exe", "/c", A.play.c_str(), nullptr};
#else
    const char *argv[] = {A.play.c_str(), nullptr};
#endif
    SDL_PropertiesID props = SDL_CreateProperties();
    SDL_SetPointerProperty(props, SDL_PROP_PROCESS_CREATE_ARGS_POINTER, (void *)argv);
    SDL_SetBooleanProperty(props, SDL_PROP_PROCESS_CREATE_BACKGROUND_BOOLEAN, true);
    SDL_Process *p = SDL_CreateProcessWithProperties(props);
    SDL_DestroyProperties(props);
    if (p) {
        SDL_DestroyProcess(p);
        quit();
    } else {
        A.err_text = "The game could not be started.", A.err_detail = A.play + "\n" + SDL_GetError(), A.err_command.clear();
        A.card = CARD_ERROR, A.card_opened = false;
    }
}

// ---- the four views of the right-hand column

static void view_form(float w, float s) {
    ImGui::PushFont(F_TITLE);
    ImGui::TextColored(TEXT, "DekooNut");
    ImGui::PopFont();
    wrapped(F_BODY, DIM, "Ocarina of Time, built on this PC from your own copy of the game.", w);
    gap(26 * s);

    bool folder = !A.game.empty() && is_dir(A.game), have_game = !A.game.empty(), have_dest = !A.dest.empty();
    heading(1, "YOUR COPY OF THE GAME", have_game, s);
    if (field("##game", have_game ? base_name(A.game) : "", "Your .nup file, update 1.1.1 included", w, s)) {
        static const SDL_DialogFileFilter filters[] = {{"Game container", "nup"}, {"All files", "*"}};
        SDL_ShowOpenFileDialog(picked_game, NULL, g_win, filters, 2, NULL, false);
    }
    static bool first = true;
    if (first) ImGui::SetItemDefaultFocus(), first = false;
    if (link(folder ? "A folder is chosen: choose another" : "Or choose the folder that holds exefs and romfs", s))
        SDL_ShowOpenFolderDialog(picked_game, NULL, g_win, NULL, false);
    gap(14 * s);

    heading(2, "WHERE TO INSTALL (ABOUT 36 GB)", have_dest, s);
    if (field("##dest", A.dest, "Choose a folder", w, s)) SDL_ShowOpenFolderDialog(picked_dest, NULL, g_win, NULL, false);
#ifdef _WIN32
    // The Start menu entry is always written; the desktop is the player's, so a shortcut there is asked for.
    ImGui::PushFont(F_SMALL);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 5 * s);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.5f * s);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(3 * s, 3 * s));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemInnerSpacing, ImVec2(10 * s, 0));
    ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(1, 0.9f, 0.75f, 0.06f));
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(1, 0.9f, 0.75f, 0.14f));
    ImGui::PushStyleColor(ImGuiCol_FrameBgActive, ImVec4(1, 0.9f, 0.75f, 0.20f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.96f, 0.74f, 0.28f, 0.40f));
    ImGui::PushStyleColor(ImGuiCol_CheckMark, GOLD);
    ImGui::PushStyleColor(ImGuiCol_Text, A.desktop ? TEXT : DIM);
    ImGui::Checkbox("Also put a shortcut on the desktop", &A.desktop);
    ImGui::PopStyleColor(6);
    ImGui::PopStyleVar(4);
    ImGui::PopFont();
    gap(8 * s);
#else
    gap(22 * s);
#endif

    heading(3, "QUALITY", !A.levels.empty(), s);
    for (size_t i = 0; i < A.levels.size(); i++) {
        if (i) ImGui::SameLine(0, 10 * s);
        bool chosen = (int)i == A.level;
        ImGui::PushFont(F_BODY);
        push_button(chosen, 23 * s, s);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(22 * s, 0));
        if (ImGui::Button(A.levels[i].name.c_str(), ImVec2(0, 46 * s))) A.level = (int)i;
        ImGui::PopStyleVar();
        pop_button();
        ImGui::PopFont();
    }
    if (!A.levels.empty()) {
        const Level &l = A.levels[A.level];
        gap(2 * s);
        wrapped(F_SMALL, DIM, l.about, w);
        if (!A.suggested.empty())
            wrapped(F_SMALL, l.id == A.suggested ? ImVec4(0.62f, 0.80f, 0.45f, 1) : ImVec4(0.90f, 0.72f, 0.36f, 1),
                    l.id == A.suggested ? "Suggested for this PC: " + A.suggested_why : "The suggestion for this PC is " + A.suggested_name + ": " + A.suggested_why, w);
    }
    gap(22 * s);

    bool ready = A.info_ready && have_game && have_dest;
    if (A.installed) {
        if (button("Play", w * 0.56f, s, true, true)) A.play = A.dest + LAUNCHER, play();
        ImGui::SameLine(0, 14 * s);
        ImGui::BeginDisabled(!ready);
        if (button("Build again", w * 0.44f - 14 * s, s, false, true)) start_build();
        ImGui::EndDisabled();
    } else {
        ImGui::BeginDisabled(!ready);
        if (button("Build the game", w, s, true, true)) start_build();
        ImGui::EndDisabled();
    }
    gap(4 * s);
    const char *status = !A.info_ready ? "Checking this PC..." : !A.missing.empty() ? "This PC needs a few build tools first: Build shows them."
                       : !have_game ? "Choose your copy of the game to begin." : !have_dest ? "Choose where to install."
                       : A.installed ? "Installed. Build again after an update of DekooNut." : "This is done once and can take an hour on a slower PC.";
    wrapped(F_SMALL, DIM, status, w);
}

static void view_build(float w, float s, float t) {
    ImGui::PushFont(F_SMALL);
    ImGui::TextColored(GOLD, A.cancelling ? "STOPPING" : A.steps.size() == 1 ? "ONE MOMENT" : "BUILDING");
    ImGui::PopFont();
    wrapped(F_HEAD, TEXT, A.cur >= 0 ? A.steps[A.cur].title : "Starting", w);
    char pct[16];
    snprintf(pct, sizeof pct, "%d%%", (int)(A.shown * 100 + 0.5f));
    ImGui::PushFont(F_TITLE);
    ImGui::TextColored(GOLD, "%s", pct);
    ImGui::PopFont();
    char info[96];
    snprintf(info, sizeof info, "Step %d of %d   ·   %s", A.cur + 1, (int)A.steps.size(), duration(SDL_GetTicks() - A.started).c_str());
    wrapped(F_BODY, DIM, A.progress.empty() ? std::string(A.steps.empty() ? "" : info) : A.progress + "   ·   " + info, w);
    gap(14 * s);
    float list_h = (A.steps.size() > 12 ? A.steps.size() : 12) * 33 * s;
    if (A.show_log) {
        ImGui::PushFont(F_MONO);
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0, 0, 0, 0.45f));
        ImGui::BeginChild("##log", ImVec2(w, list_h), 0, ImGuiWindowFlags_HorizontalScrollbar);
        for (auto &l : A.log) ImGui::TextColored(ImVec4(0.78f, 0.72f, 0.62f, 1), "%s", l.c_str());
        if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 30) ImGui::SetScrollHereY(1);
        ImGui::EndChild();
        ImGui::PopStyleColor();
        ImGui::PopFont();
    } else {
        ImVec2 p = ImGui::GetCursorScreenPos();
        ImDrawList *d = ImGui::GetWindowDrawList();
        ImGui::PushFont(F_BODY);
        for (size_t i = 0; i < A.steps.size(); i++) {
            int st = A.steps[i].state;
            float y = p.y + i * 33 * s, cy = y + ImGui::GetFontSize() / 2;
            if (st == 2) check_mark(d, ImVec2(p.x, cy - 8 * s), s * 0.75f, C(0.55f, 0.78f, 0.40f));
            else if (st == 1) d->AddCircleFilled(ImVec2(p.x + 8 * s, cy), (5.5f + 1.5f * sinf(t * 5)) * s, C(1, 0.82f, 0.40f), 16);
            else d->AddCircle(ImVec2(p.x + 8 * s, cy), 4.5f * s, C(0.50f, 0.42f, 0.32f), 16, 1.5f * s);
            d->AddText(ImVec2(p.x + 34 * s, y), ImGui::GetColorU32(st == 1 ? TEXT : st == 2 ? ImVec4(0.72f, 0.66f, 0.56f, 1) : ImVec4(0.46f, 0.40f, 0.33f, 1)), A.steps[i].title.c_str());
        }
        ImGui::PopFont();
        ImGui::Dummy(ImVec2(w, list_h));
    }
    gap(12 * s);
    float bw = (w - 14 * s) / 2;
    if (button(A.show_log ? "Show the steps" : "Show details", bw, s)) A.show_log = !A.show_log;
    if (!A.card_opened) ImGui::SetKeyboardFocusHere(-1);
    ImGui::SameLine(0, 14 * s);
    ImGui::BeginDisabled(A.cancelling);
    if (button("Cancel", bw, s)) cancel_build();
    ImGui::EndDisabled();
    gap(2 * s);
    wrapped(F_SMALL, DIM, "This only needs to be done once. A stopped build goes on from its last finished step.", w);
}

static void view_error(float w, float s) {
    ImGui::PushFont(F_SMALL);
    ImGui::TextColored(ImVec4(0.96f, 0.56f, 0.30f, 1), A.err_command.empty() ? "SETUP STOPPED" : "ONE THING FIRST");
    ImGui::PopFont();
    wrapped(F_HEAD, TEXT, A.err_text, w);
    gap(8 * s);
    wrapped(F_BODY, ImVec4(0.80f, 0.74f, 0.64f, 1), A.err_detail, w);
    gap(8 * s);
    bool has_cmd = !A.err_command.empty();
    float bw = (w - 14 * s) / 2;
    if (has_cmd) {
        ImGui::PushFont(F_MONO);
        ImVec2 p = ImGui::GetCursorScreenPos();
        float pad = 14 * s, wrap = w - 2 * pad;
        ImVec2 size = ImGui::CalcTextSize(A.err_command.c_str(), NULL, false, wrap);
        ImGui::GetWindowDrawList()->AddRectFilled(p, ImVec2(p.x + w, p.y + size.y + 2 * pad), C(0, 0, 0, 0.55f), 8 * s);
        ImGui::SetCursorScreenPos(ImVec2(p.x + pad, p.y + pad));
        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + wrap);
        ImGui::TextColored(ImVec4(0.86f, 0.90f, 0.70f, 1), "%s", A.err_command.c_str());
        ImGui::PopTextWrapPos();
        ImGui::SetCursorScreenPos(ImVec2(p.x, p.y + size.y + 2 * pad + 14 * s));
        ImGui::PopFont();
        bool just = A.copied == A.err_command && SDL_GetTicks() - A.copied_at < 1600, first = !A.card_opened;
        if (A.can_install && g_build.p == nullptr) {   // setup runs the line itself; the desktop asks for the password
            if (button("Install them now", w, s, true)) start_run(backend_args({"--install-packages", "--json"}));
            if (first) ImGui::SetKeyboardFocusHere(-1), first = false;
            gap(2 * s);
        }
        if (button(just ? "Copied" : "Copy the command", bw, s, !A.can_install)) copy_text(A.err_command);
        if (first) ImGui::SetKeyboardFocusHere(-1);
        ImGui::SameLine(0, 14 * s);
        if (button("Check again", bw, s)) A.card = CARD_NONE, ask_info();
        gap(2 * s);
        if (button("Back", bw, s)) A.card = CARD_NONE;
        return;
    }
    if (!A.log_path.empty()) {
        if (button("Open the log", bw, s)) open_path(A.log_path);
        ImGui::SameLine(0, 14 * s);
    }
    if (button("Back", bw, s)) A.card = CARD_NONE;
    if (!A.card_opened) ImGui::SetKeyboardFocusHere(-1);
}

static void view_done(float w, float s) {
    ImGui::PushFont(F_SMALL);
    ImGui::TextColored(GOLD, "DONE");
    ImGui::PopFont();
    ImGui::PushFont(F_TITLE);
    ImGui::TextColored(TEXT, "Ready to play");
    ImGui::PopFont();
    wrapped(F_BODY, DIM, (A.ended - A.started < 3000 ? std::string("Everything was up to date. ") : "Built in " + duration(A.ended - A.started) + ". ") +
                             "DekooNut is in your " MENU + (A.desktop ? " and on your desktop" : "") + ", with the icon from your game.", w);
    for (auto &warn : A.warnings) wrapped(F_SMALL, ImVec4(0.95f, 0.72f, 0.42f, 1), warn, w);
    gap(26 * s);
    if (button("Play", w, s, true, true)) play();
    if (!A.card_opened) ImGui::SetKeyboardFocusHere(-1);
    gap(4 * s);
    float bw = (w - 14 * s) / 2;
    if (button("Open the folder", bw, s)) open_path(A.dest);
    ImGui::SameLine(0, 14 * s);
    if (button("Back to setup", bw, s)) A.card = CARD_NONE;
}

static void column(float W, float H, float s, float t) {
    float x = floorf(W * 0.615f), w = floorf(W * 0.335f), y = H * (A.card == CARD_NONE ? 0.095f : 0.13f);
    // The window clips at its edge (and a pixel inside it): a margin keeps the outline of a full-width row of
    // buttons whole. The content still starts at x and is w wide.
    float m = ceilf(4 * s);
    ImGui::SetNextWindowPos(ImVec2(x - m, y));
    ImGui::SetNextWindowSize(ImVec2(w + 2 * m, H - y - 20 * s));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(m, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 9 * s));
    ImGui::Begin("##column", NULL, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoSavedSettings);
    if (A.card == CARD_NONE) view_form(w, s);
    else if (A.card == CARD_BUILD) view_build(w, s, t);
    else if (A.card == CARD_ERROR) view_error(w, s);
    else view_done(w, s);
    A.card_opened = true;
    if (A.card == CARD_NONE) {   // inside this window: one of its own would lie under it and never get the click
        ImGui::PushFont(F_SMALL);
        float qw = ImGui::CalcTextSize("Quit").x + 16 * s;
        ImGui::PopFont();
        ImGui::SetCursorScreenPos(ImVec2(x + w - qw, H - 66 * s));
        if (link("Quit", s)) quit();
    }
    ImGui::End();
    ImGui::PopStyleVar(2);

    // this PC and the version, along the foot
    ImDrawList *d = ImGui::GetForegroundDrawList();
    ImGui::PushFont(F_SMALL);
    float fy = H - 44 * s, fx = 40 * s;
    auto fact = [&](const std::string &text, bool ok) {
        if (ok) check_mark(d, ImVec2(fx, fy + 2 * s), s * 0.7f, C(0.55f, 0.78f, 0.40f));
        else d->AddCircleFilled(ImVec2(fx + 8 * s, fy + 10 * s), 5 * s, C(0.96f, 0.56f, 0.30f));
        d->AddText(ImVec2(fx + 26 * s, fy), ImGui::GetColorU32(DIM), text.c_str());
        fx += 26 * s + ImGui::CalcTextSize(text.c_str()).x + 36 * s;
    };
    if (A.info_ready) {
        fact(A.gpu.empty() ? "No NVIDIA graphics card found" : A.gpu, !A.gpu.empty());
        char b[64];
        snprintf(b, sizeof b, "%d build tools to install", (int)A.missing.size());
        fact(A.missing.empty() ? "Build tools ready" : b, A.missing.empty());
    }
    if (!A.version.empty()) d->AddText(ImVec2(fx, fy), C(0.50f, 0.44f, 0.36f), ("Version " + A.version).c_str());
    ImGui::PopFont();
}

// ---- fonts and the main loop

static void load_fonts(float s) {
    ImGuiIO &io = ImGui::GetIO();
    io.Fonts->Clear();
    std::string roboto = A.root + "/runtime/ui/fonts/Roboto-Medium.ttf";
    auto add = [&](const std::string &path, float px) {
        ImFont *f = io.Fonts->AddFontFromFileTTF(path.c_str(), floorf(px * s));
        return f ? f : io.Fonts->AddFontDefault();
    };
    F_ROW = add(roboto, 25);   // the first font is the default
    F_TITLE = add(roboto, 84);
    F_HEAD = add(roboto, 34);
    F_BODY = add(roboto, 23);
    F_SMALL = add(roboto, 19);
#ifdef _WIN32
    const char *mono = "C:\\Windows\\Fonts\\consola.ttf";
#else
    const char *mono = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf";
#endif
    SDL_PathInfo info;
    F_MONO = add(SDL_GetPathInfo(mono, &info) ? mono : roboto, 18);
    io.Fonts->Build();
    g_font_scale = s;
}

static void save_shot() {
    SDL_Surface *shot = SDL_RenderReadPixels(g_ren, NULL);
    if (shot && SDL_SavePNG(shot, A.shot.c_str())) printf("shot: %s\n", A.shot.c_str());
    else fprintf(stderr, "shot failed: %s\n", SDL_GetError());
    if (shot) SDL_DestroySurface(shot);
}

int main(int argc, char **argv) {
    int w = 0, h = 0;
    std::string backend;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i], v = i + 1 < argc ? argv[i + 1] : "";
        if (a == "--root") A.root = v, i++;
        else if (a == "--python") A.python = v, i++;
        else if (a == "--backend") backend = v, i++;
        else if (a == "--shot") A.shot = v, i++;
        else if (a == "--shot-at") A.shot_at = (float)atof(v.c_str()), i++;
        else if (a == "--size") sscanf(v.c_str(), "%dx%d", &w, &h), i++;
        else if (a == "--game") A.game = v, i++;
        else if (a == "--dest") A.dest = v, i++;
        else if (a == "--autobuild") A.autobuild = true;
        else if (a == "--click") A.click = v, i++;
    }
    if (A.root.empty() || (A.python.empty() && backend.empty())) {
        fprintf(stderr, "usage: dekoonut-setup --root DIR --python EXE (started by ./setup.sh)\n");
        return 2;
    }
    if (backend.empty()) A.backend = {A.python, "-m", "setup"};
    else
        for (size_t at = 0; at < backend.size();) {
            size_t end = backend.find(' ', at);
            if (end == std::string::npos) end = backend.size();
            if (end > at) A.backend.push_back(backend.substr(at, end - at));
            at = end + 1;
        }

    SDL_SetAppMetadata("DekooNut Setup", NULL, "dekoonut-setup");
    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) {
        fprintf(stderr, "SDL: %s\n", SDL_GetError());
        return 1;
    }
    if (!w) {
        const SDL_DisplayMode *mode = SDL_GetDesktopDisplayMode(SDL_GetPrimaryDisplay());
        int dh = mode ? mode->h : 1080;
        h = dh >= 2000 ? 1440 : dh >= 1300 ? 1080 : dh >= 1000 ? 900 : 720;
        w = h * 16 / 9;
    }
    g_win = SDL_CreateWindow("DekooNut Setup", w, h, SDL_WINDOW_RESIZABLE);
    g_ren = g_win ? SDL_CreateRenderer(g_win, NULL) : NULL;
    if (!g_ren) {
        fprintf(stderr, "SDL: %s\n", SDL_GetError());
        return 1;
    }
    SDL_SetRenderVSync(g_ren, 1);
    SDL_SetWindowMinimumSize(g_win, 960, 540);
    g_pick_lock = SDL_CreateMutex();
    if (SDL_Surface *icon = SDL_LoadPNG((A.root + "/assets/icon.png").c_str())) {
        SDL_SetWindowIcon(g_win, icon);
        SDL_DestroySurface(icon);
    }
    load_nut(A.root + "/assets/icon.png");

    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.IniFilename = NULL;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_NavEnableGamepad;
    ImGui::StyleColorsDark();
    ImGui::GetStyle().Colors[ImGuiCol_NavCursor] = ImVec4(0.93f, 0.76f, 0.33f, 0.9f);
    ImGui_ImplSDL3_InitForSDLRenderer(g_win, g_ren);
    ImGui_ImplSDLRenderer3_Init(g_ren);
    ask_info();

    Uint64 t0 = SDL_GetTicks(), last = t0;
    bool running = true, shot_taken = false;
    while (running) {
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            ImGui_ImplSDL3_ProcessEvent(&ev);
            if (ev.type == SDL_EVENT_QUIT || (ev.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED && ev.window.windowID == SDL_GetWindowID(g_win)))
                running = false;
        }
        if (SDL_GetWindowFlags(g_win) & SDL_WINDOW_MINIMIZED) {
            poll_backend();
            SDL_Delay(50);
            continue;
        }
        poll_backend();
        take_picks();
        int pw, ph;
        SDL_GetWindowSize(g_win, &pw, &ph);
        float W = (float)pw, H = (float)ph, s = H / 1080.f;
        if (W / H < 1.5f) s = W / 1620.f;   // a narrow window: the panel sets the scale
        if (fabsf(s - g_font_scale) > 0.004f) {
            load_fonts(s);
            ImGui_ImplSDLRenderer3_DestroyFontsTexture();
        }
        Uint64 now = SDL_GetTicks();
        float dt = (now - last) / 1000.f, t = (now - t0) / 1000.f;
        last = now;
        A.flash *= expf(-dt * 2.6f);
        float want = A.card == CARD_BUILD ? 0.25f + 0.75f * overall() : A.card == CARD_DONE || A.installed ? 1.0f : 0.0f;
        A.energy += (want - A.energy) * (1 - expf(-dt * 2));
        if (!A.click.empty() && t >= 1.0f) {   // a test's mouse click, as SDL would report it
            float cx = 0, cy = 0;
            sscanf(A.click.c_str(), "%f,%f", &cx, &cy);
            A.click.clear();
            SDL_Event m = {}, d = {}, u = {};
            m.type = SDL_EVENT_MOUSE_MOTION, m.motion.windowID = SDL_GetWindowID(g_win), m.motion.x = cx, m.motion.y = cy;
            d.type = SDL_EVENT_MOUSE_BUTTON_DOWN, d.button.windowID = m.motion.windowID, d.button.button = SDL_BUTTON_LEFT, d.button.down = true, d.button.x = cx, d.button.y = cy;
            u = d, u.type = SDL_EVENT_MOUSE_BUTTON_UP, u.button.down = false;
            SDL_PushEvent(&m), SDL_PushEvent(&d), SDL_PushEvent(&u);
        }
        if (A.autobuild && A.info_ready && A.card == CARD_NONE && !A.game.empty() && !A.dest.empty()) A.autobuild = false, start_build();

        ImGui_ImplSDLRenderer3_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();
        A.shown += (overall() - A.shown) * (1 - expf(-dt * 6));
        A.ring_frac = A.frac < A.ring_frac ? A.frac : A.ring_frac + (A.frac - A.ring_frac) * (1 - expf(-dt * 8));
        scene(ImGui::GetBackgroundDrawList(), W, H, s, t);
        column(W, H, s, t);
        ImGui::Render();
        SDL_SetRenderDrawColor(g_ren, 0, 0, 0, 255);
        SDL_RenderClear(g_ren);
        ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), g_ren);
        if (!A.shot.empty() && !shot_taken && t >= A.shot_at) save_shot(), shot_taken = true, running = false;
        SDL_RenderPresent(g_ren);
    }
    if (g_build.p) {   // closing the window stops the build; the next start goes on from the last finished step
        cancel_build();
        int code;
        for (int i = 0; i < 300 && !SDL_WaitProcess(g_build.p, false, &code); i++) SDL_Delay(50);
        if (!SDL_WaitProcess(g_build.p, false, &code)) g_build.cancel();
    }
    ImGui_ImplSDLRenderer3_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    SDL_DestroyRenderer(g_ren);
    SDL_DestroyWindow(g_win);
    SDL_Quit();
    return 0;
}
