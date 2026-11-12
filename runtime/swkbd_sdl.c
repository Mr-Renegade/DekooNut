// The software keyboard's front end for SDL windows (swkbd.h). There is no text entry window yet: the text entry takes
// the starting text at once, the game's initial text, else the default name from the catalog.
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "swkbd.h"

int text_entry_run(const SwkbdRequest *req, uint16_t *out, size_t cap, size_t *n) {
    size_t k = 0;
    while (k < cap && req->initial[k]) k++;
    memcpy(out, req->initial, k * 2);
    *n = k;
    rt_log("swkbd: no text entry window on this host yet: the %s name is taken", req->game_initial ? "game's" : "default");
    return k > 0;
}
