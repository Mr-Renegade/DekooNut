// Keyboard, mouse and gamepad bindings, free of any window system: tables map platform-neutral key and pad ids to Switch
// actions. input_win.c translates Win32 keys and SDL buttons to these ids; the menu's Controls tab edits a copy.
#pragma once
#include <stddef.h>
#include <stdint.h>

// Actions are the bit numbers of libnx HidNpadButton (input.h); 16..23 are the stick directions, which keys drive as
// stick deflection.
enum {
    B_A = 0, B_B, B_X, B_Y, B_STICK_L, B_STICK_R, B_L, B_R, B_ZL, B_ZR, B_PLUS, B_MINUS, B_LEFT, B_UP, B_RIGHT, B_DOWN,
    B_SL_LEFT = 16, B_SL_UP, B_SL_RIGHT, B_SL_DOWN, B_SR_LEFT, B_SR_UP, B_SR_RIGHT, B_SR_DOWN,
    ACTION_COUNT = 24
};

// Keys and mouse buttons. F1, F10, F11 and Esc belong to the menu and the window, F8 to the flight recorder: no ids.
enum {
    KEY_NONE = 0,
    KEY_A, KEY_B, KEY_C, KEY_D, KEY_E, KEY_F, KEY_G, KEY_H, KEY_I, KEY_J, KEY_K, KEY_L, KEY_M,   // letters and digits are
    KEY_N, KEY_O, KEY_P, KEY_Q, KEY_R, KEY_S, KEY_T, KEY_U, KEY_V, KEY_W, KEY_X, KEY_Y, KEY_Z,   // consecutive
    KEY_0, KEY_1, KEY_2, KEY_3, KEY_4, KEY_5, KEY_6, KEY_7, KEY_8, KEY_9,
    KEY_SPACE, KEY_TAB, KEY_LSHIFT, KEY_LCTRL, KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT,
    KEY_MOUSE_L, KEY_MOUSE_R, KEY_MOUSE_M, KEY_MOUSE_X1, KEY_MOUSE_X2,   // the mouse buttons are consecutive
    KEY_ENTER, KEY_BACKSPACE, KEY_RSHIFT, KEY_RCTRL, KEY_LALT, KEY_CAPSLOCK, KEY_INSERT, KEY_DELETE, KEY_HOME, KEY_END,
    KEY_PAGEUP, KEY_PAGEDOWN,
    KEY_F2, KEY_F3, KEY_F4, KEY_F5, KEY_F6, KEY_F7, KEY_F9, KEY_F12,
    KEY_NUM0, KEY_NUM1, KEY_NUM2, KEY_NUM3, KEY_NUM4, KEY_NUM5, KEY_NUM6, KEY_NUM7, KEY_NUM8, KEY_NUM9,   // consecutive
    KEY_NUM_ADD, KEY_NUM_SUB, KEY_NUM_MUL, KEY_NUM_DIV, KEY_NUM_DECIMAL,
    // the punctuation keys of a US keyboard, by position; the character on the key follows the keyboard layout
    KEY_MINUS, KEY_EQUALS, KEY_LBRACKET, KEY_RBRACKET, KEY_SEMICOLON, KEY_APOSTROPHE, KEY_COMMA, KEY_PERIOD, KEY_SLASH,
    KEY_BACKSLASH, KEY_GRAVE,
    KEY_COUNT
};

// Gamepad inputs by position (Nintendo layout, as SDL names them). ZL and ZR are the analog triggers. The paddles are
// back buttons (SDL's LEFT_PADDLE1 and so on); no Switch button has them by default.
enum {
    PAD_NONE = 0,
    PAD_SOUTH, PAD_EAST, PAD_WEST, PAD_NORTH, PAD_LSTICK, PAD_RSTICK, PAD_L, PAD_R, PAD_START, PAD_BACK,
    PAD_DPAD_LEFT, PAD_DPAD_UP, PAD_DPAD_RIGHT, PAD_DPAD_DOWN, PAD_ZL, PAD_ZR,
    PAD_LPADDLE1, PAD_RPADDLE1, PAD_LPADDLE2, PAD_RPADDLE2,
    PAD_COUNT
};

// Gamepad layouts (OOT_PAD_LAYOUT): which face button is Switch A. Nintendo: by position, east A, south B, west Y,
// north X. Xbox: south A, east B, west X, north Y, as an Xbox pad labels them (cross A, circle B on a PlayStation pad).
enum { PAD_LAYOUT_NINTENDO, PAD_LAYOUT_XBOX, PAD_LAYOUT_COUNT };

enum { INPUT_BINDING_MAX = 96, INPUT_PAD_BINDING_MAX = 48 };

typedef struct { uint8_t key; uint8_t action; } Binding;
typedef struct { uint8_t pad; uint8_t action; } PadBinding;
// The keyboard and mouse table and one gamepad table per layout.
typedef struct InputMap {
    Binding keys[INPUT_BINDING_MAX];
    int nkeys;
    PadBinding pads[PAD_LAYOUT_COUNT][INPUT_PAD_BINDING_MAX];
    int npads[PAD_LAYOUT_COUNT];
} InputMap;

// The tables in effect. Defaults are the built-in layouts; input_map_reset restores them.
extern InputMap input_map;

void input_map_reset(void);
// The script and oot.ini names, any case when parsed; NULL for an id without one. Actions: A B X Y LCLICK RCLICK L R ZL
// ZR PLUS MINUS LEFT UP RIGHT DOWN, then LSTICK_LEFT LSTICK_UP LSTICK_RIGHT LSTICK_DOWN and the same for RSTICK (the
// stick directions the keys drive). Keys: a-z 0-9 space tab lshift ... (key_names in input_map.c). Pads: south east ...
const char *input_action_name(int action);
const char *input_key_name(int key);
const char *input_pad_name(int pad);
// Replaces every binding of the action `button` with `list`: comma separated "pad:<name>" and "key:<name>" entries,
// "key:default" / "pad:default" for that part's defaults, or "none". A part the list does not name is left unbound. Pad
// names are physical buttons, so pad entries go into every layout's table; stick directions take keys only. Invalid
// entries are logged and skipped; when none is valid the action keeps its bindings. 1 when applied.
int input_map_bind(const char *button, const char *list);
// Applies the OOT_BIND_<action> settings (cfg_get) over the defaults: at start, and again when one of them changes.
void input_map_configure(void);

// The layout for a OOT_PAD_LAYOUT value: "nintendo" or "xbox"; "xbox_prompts" is the Nintendo layout with the A/B and X/Y
// prompts swapped (romfs.c overlay_init). Anything else picks Nintendo when the pad's east button is labelled A, else Xbox.
int input_pad_layout_pick(const char *setting, int east_labelled_a);
// down[pad] != 0 for the gamepad inputs held. Returns the Switch button bits (0..15) the layout's pad table presses.
uint64_t input_map_pad_buttons(const uint8_t down[PAD_COUNT], int layout);

// down[key] != 0 for the keys held. Returns the Switch button bits (0..15) the table presses.
uint64_t input_map_buttons(const uint8_t down[KEY_COUNT]);
// out = x, y of the stick (0 left, 1 right) from its four direction actions; up is +y; a diagonal is scaled to the
// unit circle.
void input_map_stick(const uint8_t down[KEY_COUNT], int stick, int32_t out[2]);
// Mouse motion in raw counts over dt_s seconds to right-stick deflection: full deflection at 3000 / sens counts/s.
// Clamped to the unit circle, up (negative dy) is +y.
void input_mouse_stick(int32_t dx, int32_t dy, double dt_s, float sens, int32_t out[2]);
// A gamepad stick (x, y in -32768..32767, up is +y) through a round dead zone of `dead` (fraction of full deflection):
// the rest of the travel is scaled so full deflection still reaches full. The result stays inside the unit square.
void input_pad_stick(int32_t x, int32_t y, float dead, int32_t out[2]);

// ---- the remapping rows' edits, on a copy of input_map. Pure.
// action's keys (table order) / its gamepad buttons in layout, at most max; returns how many.
int input_map_keys_of(const InputMap *m, int action, uint8_t *out, int max);
int input_map_pads_of(const InputMap *m, int action, int layout, uint8_t *out, int max);
// Key slot (0 or 1) of action becomes key; KEY_NONE clears the slot. A key another action has moves here: returns that
// action, -1 when none lost it.
int input_map_set_key(InputMap *m, int action, int slot, int key);
// action's gamepad button in layout becomes pad; PAD_NONE clears it. The action that had pad takes action's old button
// (a swap): returns it, -1 when none. Remapped buttons are physical, so the face buttons and the changed actions get the
// result in every layout; a change that puts every button of layout at its default restores every layout's defaults.
int input_map_set_pad(InputMap *m, int action, int layout, int pad);
// The bind_ value that gives action its bindings of m: "" when they are the defaults (no line needed), else the entries,
// with key:default / pad:default for a part at its defaults, or "none".
void input_map_line(const InputMap *m, int action, char *out, size_t n);
