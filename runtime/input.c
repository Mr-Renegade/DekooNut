// Host input for nn::hid (ov_input.c): an SDL3 gamepad, the keyboard and the mouse merged into Switch npad button bits and
// stick axes. Button order follows libnx HidNpadButton (ISC). Bindings: input_map.c. Keyboard and mouse come from the window
// back end (input_wsys.h). Every SDL call on the gamepad is made under `lock`.
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <SDL3/SDL.h>

#include "cfg.h"
#include "host.h"
#include "input.h"
#include "input_map.h"
#include "input_wsys.h"
#include "nvn_api.h"
#include "ui/menu.h"
#include "vk.h"
#include "platform/platform.h"

enum { STICK_MAX = 32767, STICK_DIR = 16384, TRIGGER_ON = 16384, RESCAN_MS = 1000 };

static PlatMutex lock = PLAT_MUTEX_INIT;
static int state;                    // 0 not started, 1 running, -1 disabled or SDL failed
static SDL_Gamepad *pad;
static uint64_t next_scan;
static int pad_rumble;               // the open gamepad has rumble motors
static volatile int32_t pad_labels[4];  // ... and the labels of its south, east, west, north buttons (PAD_LABEL_*), any thread
static int pad_east_a;               // ... and labels its east face button A (Nintendo layout under OOT_PAD_LAYOUT=auto)
static int pad_layout = -1;          // PAD_LAYOUT_* in effect for the open gamepad, -1 to pick again; live (cfg_version)
static unsigned pad_layout_seen;
static volatile int32_t pad_layout_shown = -1;   // pad_layout for the menu (any thread), -1 without a gamepad
static SDL_JoystickID sensor_pad;    // the gamepad whose sensors the motion thread set up, 0 none
static int sensor_ok;                // ... and it has both sensors, enabled
static int motion_thread;            // 0 not started, 1 running, -1 off (OOT_MOTION=0, no SDL)

// SDL buttons of the gamepad inputs (input_map.h PAD_*), by position; PAD_ZL and PAD_ZR are the triggers (axes).
static const SDL_GamepadButton pad_sdl[PAD_COUNT] = {
    [PAD_SOUTH] = SDL_GAMEPAD_BUTTON_SOUTH, [PAD_EAST] = SDL_GAMEPAD_BUTTON_EAST, [PAD_WEST] = SDL_GAMEPAD_BUTTON_WEST,
    [PAD_NORTH] = SDL_GAMEPAD_BUTTON_NORTH, [PAD_LSTICK] = SDL_GAMEPAD_BUTTON_LEFT_STICK,
    [PAD_RSTICK] = SDL_GAMEPAD_BUTTON_RIGHT_STICK, [PAD_L] = SDL_GAMEPAD_BUTTON_LEFT_SHOULDER,
    [PAD_R] = SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, [PAD_START] = SDL_GAMEPAD_BUTTON_START, [PAD_BACK] = SDL_GAMEPAD_BUTTON_BACK,
    [PAD_DPAD_LEFT] = SDL_GAMEPAD_BUTTON_DPAD_LEFT, [PAD_DPAD_UP] = SDL_GAMEPAD_BUTTON_DPAD_UP,
    [PAD_DPAD_RIGHT] = SDL_GAMEPAD_BUTTON_DPAD_RIGHT, [PAD_DPAD_DOWN] = SDL_GAMEPAD_BUTTON_DPAD_DOWN,
    [PAD_LPADDLE1] = SDL_GAMEPAD_BUTTON_LEFT_PADDLE1, [PAD_RPADDLE1] = SDL_GAMEPAD_BUTTON_RIGHT_PADDLE1,
    [PAD_LPADDLE2] = SDL_GAMEPAD_BUTTON_LEFT_PADDLE2, [PAD_RPADDLE2] = SDL_GAMEPAD_BUTTON_RIGHT_PADDLE2,
};

static float mouse_sens = -1;              // OOT_MOUSE_SENS, 0 = mouse camera off; live (cfg_version)
static unsigned mouse_sens_seen;

static float mouse_sens_get(void) {
    unsigned ver = cfg_version();
    if (mouse_sens >= 0 && ver == mouse_sens_seen) return mouse_sens;
    mouse_sens_seen = ver;
    double v = cfg_double("OOT_MOUSE_SENS");
    mouse_sens = (float)(v < 0 ? 0 : v > 10 ? 10 : v);
    return mouse_sens;
}


// The mouse camera as a right-stick sample, held for the whole presented frame. Under `lock`.
static void mouse_camera(int32_t out[2]) {
    static uint64_t frame = ~0ull, t0;
    static int32_t held[2];
    uint64_t now = plat_now_ns();
    int32_t dx, dy;
    if (!ikb_captured()) {
        ikb_mouse_take(&dx, &dy);
        held[0] = held[1] = 0;
        frame = ~0ull;
    } else {
        uint64_t fr = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
        double age = frame == ~0ull ? 0 : (double)(now - t0) / 1e9;
        if (fr != frame) {
            double dt = age < 0.004 ? 0.004 : age > 0.1 ? 0.1 : age;
            ikb_mouse_take(&dx, &dy);
            input_mouse_stick(dx, dy, dt, mouse_sens_get(), held);
            frame = fr;
            t0 = now;
        } else if (age > 0.1) {
            held[0] = held[1] = 0;   // frames stopped: do not hold a deflection
        }
    }
    out[0] = held[0];
    out[1] = held[1];
}

static float stick_dead = -1;              // OOT_STICK_DEADZONE as a fraction of full deflection; live (cfg_version)
static unsigned stick_dead_seen;

static float stick_dead_get(void) {
    unsigned ver = cfg_version();
    if (stick_dead >= 0 && ver == stick_dead_seen) return stick_dead;
    stick_dead_seen = ver;
    double v = cfg_double("OOT_STICK_DEADZONE");
    stick_dead = (float)(v < 0 ? 0 : v > 50 ? 50 : v) / 100.f;
    return stick_dead;
}

// One gamepad stick through the dead zone; SDL's Y grows downwards, the Switch's upwards.
static void stick(SDL_GamepadAxis ax, SDL_GamepadAxis ay, float dead, int32_t out[2]) {
    input_pad_stick(SDL_GetGamepadAxis(pad, ax), -(int32_t)SDL_GetGamepadAxis(pad, ay), dead, out);
}

// A gamepad made through uinput (a virtual Xbox 360 pad): its event node sits under /sys/devices/virtual/input. Bluetooth
// (virtual/misc/uhid) and USB pads do not count; vendor and product ids cannot tell them apart.
static int pad_is_virtual(SDL_JoystickID id) {
#ifdef __linux__
    const char *path = SDL_GetGamepadPathForID(id);
    const char *ev = path ? strrchr(path, '/') : NULL;
    char link[128], real[PATH_MAX];
    if (!ev || strncmp(ev, "/event", 6)) return 0;
    snprintf(link, sizeof link, "/sys/class/input%s/device", ev);
    return realpath(link, real) && strstr(real, "/devices/virtual/input/") != NULL;
#else
    (void)id;
    return 0;
#endif
}

// The gamepad to open: the first that is not a uinput pad, else the first.
static SDL_JoystickID pad_pick(int *is_virtual) {
    int n = 0;
    SDL_JoystickID *ids = SDL_GetGamepads(&n), pick = 0;
    *is_virtual = 0;
    for (int i = 0; ids && i < n; i++)
        if (!pad_is_virtual(ids[i])) { pick = ids[i]; break; }
    if (!pick && ids && n) pick = ids[0], *is_virtual = pad_is_virtual(ids[0]);
    SDL_free(ids);
    return pick;
}

// Open a gamepad when none is open or the open one went away; while the open one is a uinput pad, a real one that
// connects takes over (the caller looks every RESCAN_MS).
static int pad_virtual;
static int pad_takeovers;   // in a row without a real pad open after: one that does not open stops being tried after 5
static void scan_pads(void) {
    if (pad && SDL_GamepadConnected(pad)) {
        int v;
        if (!pad_virtual || pad_takeovers >= 5 || !pad_pick(&v) || v) return;
        pad_takeovers++;
        rt_log("input: a real gamepad connected: it replaces the virtual one");
        SDL_CloseGamepad(pad); pad = NULL; sensor_pad = 0; sensor_ok = 0;
        plat_xchg32(&pad_layout_shown, -1);
        for (int i = 0; i < 4; i++) plat_xchg32(&pad_labels[i], PAD_LABEL_NONE);
    }
    if (pad) {
        SDL_CloseGamepad(pad); pad = NULL; sensor_pad = 0; sensor_ok = 0;
        plat_xchg32(&pad_layout_shown, -1);
        for (int i = 0; i < 4; i++) plat_xchg32(&pad_labels[i], PAD_LABEL_NONE);
        rt_log("input: gamepad disconnected");
    }
    SDL_JoystickID id = pad_pick(&pad_virtual);
    if (id) pad = SDL_OpenGamepad(id);
    if (!pad && id) {  // the picked one does not open: any that does
        int n = 0;
        SDL_JoystickID *ids = SDL_GetGamepads(&n);
        for (int i = 0; ids && i < n && !pad; i++)
            if ((pad = SDL_OpenGamepad(ids[i]))) pad_virtual = pad_is_virtual(ids[i]);
        SDL_free(ids);
    }
    if (!pad) return;
    if (!pad_virtual) pad_takeovers = 0;
    pad_rumble = SDL_GetBooleanProperty(SDL_GetGamepadProperties(pad), SDL_PROP_GAMEPAD_CAP_RUMBLE_BOOLEAN, false);
    pad_east_a = SDL_GetGamepadButtonLabel(pad, SDL_GAMEPAD_BUTTON_EAST) == SDL_GAMEPAD_BUTTON_LABEL_A;
    for (int i = 0; i < 4; i++) {
        SDL_GamepadButtonLabel lb = SDL_GetGamepadButtonLabel(pad, pad_sdl[PAD_SOUTH + i]);
        plat_xchg32(&pad_labels[i], lb == SDL_GAMEPAD_BUTTON_LABEL_A          ? PAD_LABEL_A
                                            : lb == SDL_GAMEPAD_BUTTON_LABEL_B        ? PAD_LABEL_B
                                            : lb == SDL_GAMEPAD_BUTTON_LABEL_X        ? PAD_LABEL_X
                                            : lb == SDL_GAMEPAD_BUTTON_LABEL_Y        ? PAD_LABEL_Y
                                            : lb == SDL_GAMEPAD_BUTTON_LABEL_CROSS    ? PAD_LABEL_CROSS
                                            : lb == SDL_GAMEPAD_BUTTON_LABEL_CIRCLE   ? PAD_LABEL_CIRCLE
                                            : lb == SDL_GAMEPAD_BUTTON_LABEL_SQUARE   ? PAD_LABEL_SQUARE
                                            : lb == SDL_GAMEPAD_BUTTON_LABEL_TRIANGLE ? PAD_LABEL_TRIANGLE
                                                                                      : PAD_LABEL_NONE);
    }
    pad_layout = -1;
    int motion = SDL_GamepadHasSensor(pad, SDL_SENSOR_GYRO) && SDL_GamepadHasSensor(pad, SDL_SENSOR_ACCEL);
    rt_log("input: gamepad \"%s\"%s (%s, %s, east button %s)", SDL_GetGamepadName(pad),
           pad_virtual ? ", a virtual (uinput) pad: a real one that connects replaces it" : "",
           pad_rumble ? "rumble" : "no rumble", motion ? "gyro and accelerometer" : "no motion sensors",
           pad_east_a ? "labelled A" : "not labelled A");
}

// Under lock: the layout of the open gamepad, picked again when a setting or the gamepad changed; logs each change.
static int pad_layout_get(void) {
    unsigned ver = cfg_version();
    if (pad_layout >= 0 && ver == pad_layout_seen) return pad_layout;
    pad_layout_seen = ver;
    int was = pad_layout;
    const char *set = cfg_value("OOT_PAD_LAYOUT");
    pad_layout = input_pad_layout_pick(set, pad_east_a);
    if (pad_layout != was)
        rt_log("input: gamepad layout %s (pad_layout = %s)", pad_layout == PAD_LAYOUT_XBOX ? "xbox" : "nintendo", set);
    plat_xchg32(&pad_layout_shown, pad_layout);
    return pad_layout;
}

// The OOT_BIND_* values last applied, one per line; the tables are rebuilt when one changes. Under lock.
static unsigned binds_seen = ~0u;
static char binds_applied[4096];

static void binds_refresh(void) {
    unsigned v = cfg_version();
    if (v == binds_seen) return;
    binds_seen = v;
    char now[sizeof binds_applied];
    size_t len = 0;
    for (int a = 0; a < ACTION_COUNT; a++) {
        char var[32];
        snprintf(var, sizeof var, "OOT_BIND_%s", input_action_name(a));
        const char *s = cfg_get(var);
        len += (size_t)snprintf(now + len, len < sizeof now ? sizeof now - len : 0, "%s\n", s ? s : "");
    }
    if (!strcmp(now, binds_applied)) return;
    memcpy(binds_applied, now, sizeof now);
    input_map_configure();
}

static void start(void) {
    binds_refresh();
    if (!SDL_Init(SDL_INIT_GAMEPAD)) { state = -1; rt_log("input: SDL gamepad init failed (%s), keyboard only", SDL_GetError()); }
    else state = 1;
    if (state == 1) {
        scan_pads();
        if (!pad) rt_log("input: no gamepad (keyboard only until one is connected)");
    }
    if (state == -1) state = 2;   // keyboard only
}

static void stick_bits(uint64_t *b, int32_t x, int32_t y, int first) {
    if (x < -STICK_DIR) *b |= 1ull << (first + 0);
    if (y > STICK_DIR) *b |= 1ull << (first + 1);
    if (x > STICK_DIR) *b |= 1ull << (first + 2);
    if (y < -STICK_DIR) *b |= 1ull << (first + 3);
}

static void pick(int32_t *x, int32_t *y, int32_t kx, int32_t ky) {
    int64_t a = (int64_t)*x * *x + (int64_t)*y * *y, b = (int64_t)kx * kx + (int64_t)ky * ky;
    if (b > a) { *x = kx; *y = ky; }
}

// Under lock, state 1: the open gamepad's inputs by position (pd[PAD_*] != 0 held); 0 without a connected gamepad.
static int pad_physical(uint8_t pd[PAD_COUNT]) {
    if ((!pad || pad_virtual || !SDL_GamepadConnected(pad)) && plat_now_ns() / 1000000u >= next_scan) { next_scan = plat_now_ns() / 1000000u + RESCAN_MS; scan_pads(); }
    SDL_UpdateGamepads();
    if (!pad || !SDL_GamepadConnected(pad)) return 0;
    for (int p = PAD_SOUTH; p < PAD_ZL; p++) pd[p] = SDL_GetGamepadButton(pad, pad_sdl[p]);
    pd[PAD_ZL] = SDL_GetGamepadAxis(pad, SDL_GAMEPAD_AXIS_LEFT_TRIGGER) > TRIGGER_ON;
    pd[PAD_ZR] = SDL_GetGamepadAxis(pad, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER) > TRIGGER_ON;
    for (int p = PAD_ZR + 1; p < PAD_COUNT; p++) pd[p] = SDL_GetGamepadButton(pad, pad_sdl[p]);   // false without paddles
    return 1;
}

// Under lock, state 1: the open gamepad's buttons (Switch bits after OOT_PAD_LAYOUT and the OOT_BIND_* lines) and
// sticks; 0 without a connected gamepad.
static int pad_read(uint64_t *b, int32_t sl[2], int32_t sr[2]) {
    uint8_t pd[PAD_COUNT] = {0};
    if (!pad_physical(pd)) return 0;
    *b |= input_map_pad_buttons(pd, pad_layout_get());
    float dead = stick_dead_get();
    stick(SDL_GAMEPAD_AXIS_LEFTX, SDL_GAMEPAD_AXIS_LEFTY, dead, sl);
    stick(SDL_GAMEPAD_AXIS_RIGHTX, SDL_GAMEPAD_AXIS_RIGHTY, dead, sr);
    return 1;
}

static void poll_merged(uint64_t *buttons, int32_t *lx, int32_t *ly, int32_t *rx, int32_t *ry) {
    uint64_t b = 0;
    int32_t sl[2] = {0, 0}, sr[2] = {0, 0};
    plat_mutex_lock(&lock);
    if (!state) start();
    if (state > 0) {
        binds_refresh();
        if (state == 1) pad_read(&b, sl, sr);
        if (ikb_foreground()) {
            uint8_t down[KEY_COUNT] = {0};
            // Mouse buttons count only while the cursor is captured, and a button already held when the capture began
            // (the click that refocused the window) is ignored until it is released.
            static int was_captured;
            static uint8_t mouse_ignore[KEY_COUNT];
            int cap = ikb_captured();
            for (int k = 1; k < KEY_COUNT; k++) {
                down[k] = ikb_down(k);
                if (k < KEY_MOUSE_L || k > KEY_MOUSE_X2) continue;
                if (cap && !was_captured) mouse_ignore[k] = down[k];
                if (!down[k]) mouse_ignore[k] = 0;
                if (!cap || mouse_ignore[k]) down[k] = 0;
            }
            was_captured = cap;
            b |= input_map_buttons(down);
            int32_t k[2];
            input_map_stick(down, 0, k);
            pick(&sl[0], &sl[1], k[0], k[1]);
            input_map_stick(down, 1, k);
            pick(&sr[0], &sr[1], k[0], k[1]);
            mouse_camera(k);
            pick(&sr[0], &sr[1], k[0], k[1]);
        }
    }
    // Minus and Plus held together (gamepad or keys) open or close the menu and never reach the game
    static int combo_held;
    int32_t st[4] = {sl[0], sl[1], sr[0], sr[1]};
    int toggle = input_menu_gate(&combo_held, menu_is_open(), &b, st);
    static int overlay_held;   // L3+R3 step the performance overlay (F10)
    int overlay = input_overlay_gate(&overlay_held, &b);
    sl[0] = st[0], sl[1] = st[1], sr[0] = st[2], sr[1] = st[3];
    plat_mutex_unlock(&lock);
    if (toggle) menu_toggle();
    if (overlay) menu_perf_cycle();
    stick_bits(&b, sl[0], sl[1], B_SL_LEFT);
    stick_bits(&b, sr[0], sr[1], B_SR_LEFT);
    *buttons = b; *lx = sl[0]; *ly = sl[1]; *rx = sr[0]; *ry = sr[1];
}

// Text entry (swkbd_*.c) blocks the game's input while open and until every button and stick is released after it closes.
// g_block: 0 off, 1 open, 2 closed, waiting for the release. The cursor is released while it is open.
static volatile int32_t g_block;
static volatile int32_t g_menu;   // the in-game menu, the same way (input_menu)

int input_cursor_allowed(void) {
    return g_block != 1 && g_menu != 1 && mouse_sens_get() > 0;
}

void input_block(int on) {
    plat_xchg32(&g_block, on ? 1 : 2);
    ikb_cursor_update();
}

// The in-game menu (ui/menu.cpp) blocks input the same way (g_menu); the cursor is released while it is open.
void input_menu(int open) {
    plat_xchg32(&g_menu, open ? 1 : 2);
    if (!open) ikb_menu_closed();   // the capture before Esc comes back
    ikb_cursor_update();
}

// 1 while flag (g_block, g_menu) blocks the game's input; state 2 ends at the first poll with nothing held.
static int blocked(volatile int32_t *flag, uint64_t buttons) {
    int32_t k = *flag;
    return k && !(k == 2 && !buttons && plat_cas32(flag, 0, 2) == 2);
}

void input_poll(uint64_t *buttons, int32_t *lx, int32_t *ly, int32_t *rx, int32_t *ry) {
    poll_merged(buttons, lx, ly, rx, ry);
    int text = blocked(&g_block, *buttons), menu = blocked(&g_menu, *buttons);
    if (!text && !menu) return;
    *buttons = 0; *lx = *ly = *rx = *ry = 0;
}

void input_poll_ui(uint64_t *buttons, int32_t *lx, int32_t *ly, int32_t *rx, int32_t *ry) {
    poll_merged(buttons, lx, ly, rx, ry);
}

int input_pad_read(uint64_t *buttons, int32_t *lx, int32_t *ly) {
    uint64_t b = 0;
    int32_t sl[2] = {0, 0}, sr[2] = {0, 0};
    plat_mutex_lock(&lock);
    if (!state) start();
    if (state > 0) binds_refresh();
    int ok = state == 1 && pad_read(&b, sl, sr);
    plat_mutex_unlock(&lock);
    *buttons = b;
    *lx = sl[0];
    *ly = sl[1];
    return ok;
}

uint16_t rumble_motor(float a, float curve, float strength) {
    if (!(a > 0) || !(strength > 0)) return 0;
    float v = powf(a > 1 ? 1 : a, curve) * strength * 65535.0f + 0.5f;
    return v >= 65535.0f ? 65535 : (uint16_t)v;
}

int input_pad_layout(void) {
    return (int)plat_cas32(&pad_layout_shown, 0, 0);
}

int input_pad_label(int pad_id) {
    if (pad_id < PAD_SOUTH || pad_id > PAD_NORTH) return PAD_LABEL_NONE;
    return (int)plat_cas32(&pad_labels[pad_id - PAD_SOUTH], 0, 0);
}

void input_map_snapshot(InputMap *m) {
    plat_mutex_lock(&lock);
    if (!state) start();
    if (state > 0) binds_refresh();
    *m = input_map;
    plat_mutex_unlock(&lock);
}

int input_capture(int reset, int *id) {
    static uint8_t prev_k[KEY_COUNT], prev_p[PAD_COUNT];
    static int prev_esc;
    uint8_t k[KEY_COUNT] = {0}, p[PAD_COUNT] = {0};
    int fg = ikb_foreground();
    for (int i = 1; i < KEY_COUNT; i++) k[i] = fg && ikb_down(i);
    int esc = fg && ikb_escape();
    plat_mutex_lock(&lock);
    if (!state) start();
    if (state == 1) pad_physical(p);
    plat_mutex_unlock(&lock);
    int got = INPUT_CAPTURE_NONE;
    if (!reset) {
        if (esc && !prev_esc) got = INPUT_CAPTURE_CANCEL;
        for (int i = 1; i < KEY_COUNT && !got; i++)
            if (k[i] && !prev_k[i]) got = INPUT_CAPTURE_KEY, *id = i;
        for (int i = 1; i < PAD_COUNT && !got; i++)
            if (p[i] && !prev_p[i]) got = INPUT_CAPTURE_PAD, *id = i;
    }
    memcpy(prev_k, k, sizeof k);
    memcpy(prev_p, p, sizeof p);
    prev_esc = esc;
    return got;
}

int input_rumble_triggers_enabled(void) {
    return cfg_bool("OOT_RUMBLE_TRIGGERS");
}

int input_rumble(uint16_t low, uint16_t high, uint32_t ms) {
    int ok = 0;
    static int triggers = -1;
    static unsigned seen;
    if (triggers < 0 || seen != cfg_version()) seen = cfg_version(), triggers = input_rumble_triggers_enabled();
    plat_mutex_lock(&lock);
    if (state == 1 && pad && pad_rumble) {
        ok = SDL_RumbleGamepad(pad, low, high, ms);
        if (triggers) SDL_RumbleGamepadTriggers(pad, high, high, ms);   // unsupported pads return false: ignored
    }
    plat_mutex_unlock(&lock);
    return ok;
}

// SDL sensor axes: x right, y out of the face, z toward the player; accelerometer in m/s^2 (flat: y = +9.81), gyro in rad/s.
// Switch six-axis: x right, y away from the player, z out of the face; acceleration in G (flat: z = -1). So Switch x = SDL x,
// y = -SDL z, z = SDL y; the gyro keeps its sense, the acceleration is negated and scaled to G.
void input_motion_axes(const float a[3], const float g[3], double acc[3], double gyro[3]) {
    acc[0] = -a[0] / SDL_STANDARD_GRAVITY;
    acc[1] = a[2] / SDL_STANDARD_GRAVITY;
    acc[2] = -a[1] / SDL_STANDARD_GRAVITY;
    gyro[0] = g[0];
    gyro[1] = -g[2];
    gyro[2] = g[1];
}

// Under lock: (re)enable the sensors when the open gamepad changed; 1 if it has both and they are on.
static int sensors_ready(void) {
    SDL_JoystickID id = SDL_GetGamepadID(pad);
    if (id == sensor_pad) return sensor_ok;
    sensor_pad = id;
    // Nothing drains the SDL event queue: no event per sample (the data still updates, SDL_SendJoystickSensor)
    SDL_SetEventEnabled(SDL_EVENT_GAMEPAD_SENSOR_UPDATE, false);
    sensor_ok =SDL_GamepadHasSensor(pad, SDL_SENSOR_GYRO) && SDL_GamepadHasSensor(pad, SDL_SENSOR_ACCEL) &&
                SDL_SetGamepadSensorEnabled(pad, SDL_SENSOR_GYRO, true) &&
                SDL_SetGamepadSensorEnabled(pad, SDL_SENSOR_ACCEL, true);
    if (sensor_ok) rt_log("input: motion sensors on (gyro %.0f Hz, accelerometer %.0f Hz)",
                          (double)SDL_GetGamepadSensorDataRate(pad, SDL_SENSOR_GYRO),
                          (double)SDL_GetGamepadSensorDataRate(pad, SDL_SENSOR_ACCEL));
    else rt_log("input: \"%s\" has no gyro and accelerometer: the six-axis sensor reads a controller at rest",
                SDL_GetGamepadName(pad));
    return sensor_ok;
}

int input_motion_enabled(void) {
    static int on = -1;
    static unsigned seen;
    unsigned v = cfg_version();
    if (on < 0 || v != seen) {
        seen = v;
        on = cfg_bool("OOT_MOTION");
    }
    return on;
}

// 250 Hz: updates the gamepads (reports, rumble expiry) and feeds the newest gyro and accelerometer sample into the
// fusion; without sensors or with OOT_MOTION=0 the six-axis sensor reads at rest.
static void motion_main(void *arg) {
    (void)arg;
    for (;;) {
        plat_wait_ns(4000000);
        float a[3], g[3];
        int have = 0;
        plat_mutex_lock(&lock);
        if (pad && SDL_GamepadConnected(pad)) {
            SDL_UpdateGamepads();
            have = sensors_ready() && SDL_GetGamepadSensorData(pad, SDL_SENSOR_ACCEL, a, 3) &&
                   SDL_GetGamepadSensorData(pad, SDL_SENSOR_GYRO, g, 3);
        }
        plat_mutex_unlock(&lock);
        if (!have || !input_motion_enabled()) continue;
        double acc[3], gyro[3];
        input_motion_axes(a, g, acc, gyro);
        motion_feed(motion_global(), acc, gyro, kern_now_ns());
    }
}

void input_motion_start(void) {
    plat_mutex_lock(&lock);
    if (!state) start();
    if (!motion_thread) {
        if (!input_motion_enabled()) rt_log("input: motion off (OOT_MOTION=0); it can be turned on while the game runs");
        if (state != 1) motion_thread = -1;
        else {
            if (!plat_thread_spawn(motion_main, NULL, "motion", 0)) rt_fatal("input: cannot start the motion thread");
            motion_thread = 1;
        }
    }
    plat_mutex_unlock(&lock);
}
