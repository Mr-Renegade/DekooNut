# SPDX-License-Identifier: GPL-3.0-or-later
"""The quality levels setup offers, heaviest first. A level is a game-data pack (tools/build_gamedata.py) and the
environment the launcher starts the game with; nothing else differs between two installs.

    id: {'name': shown in the window, 'about': one sentence, 'pack': pack name or None, 'mods': the pack's recipe,
         'env': {OOT_*: value} the launcher exports, 'ini': {key: value} written to the install's oot.ini (settings
         the player may change there later; setup leaves a changed one alone),
         'needs': {'vram_gb': video memory, 'ram_gb': memory the level is measured to need}}

suggest() picks the heaviest level this PC meets; the window preselects it and says why.
"""

LEVELS = {
    'default': {
        'name': 'Default',
        'about': 'The settings the game is played and tested with: 4K output through DLSS at 60 fps.',
        'pack': 'hs625', 'mods': ['dynres:0.625', 'heap:208'], 'env': {},
        'ini': {'romfs_mods': 'gamedata/hs625', 'gfx_scale': '0'}, 'needs': {'vram_gb': 10, 'ram_gb': 16},
    },
    # Measured on an RTX 3070 (8 GB) with a Ryzen 5 9600X at 2560x1440, Hyrule Field and Kakariko test saves
    # (docs/quality-levels.md): Default 41 fps there; Balanced 55 fps standing in the field, 60 running through it and
    # in the village; Light 60 everywhere. Both hold the game's 6 GB video-memory budget on that card.
    'balanced': {
        'name': 'Balanced',
        'about': "For 8 GB cards such as an RTX 3070: the game's own 1280x720 scene, the lighter DLSS model and no "
                 'grass blades. 55 to 60 fps at 2560x1440 on that card.',
        'pack': None, 'mods': [], 'env': {},
        'ini': {'gfx_scale': '2', 'dlss': 'fast', 'grass': 'off'}, 'needs': {'vram_gb': 8, 'ram_gb': 16},
    },
    'light': {
        'name': 'Light',
        'about': 'Balanced without DLSS: FSR scales the picture. 60 fps at 2560x1440 on an RTX 3070; the choice for '
                 'cards without DLSS.',
        'pack': None, 'mods': [], 'env': {},
        'ini': {'gfx_scale': '2', 'dlss': 'off', 'grass': 'off'}, 'needs': {'vram_gb': 8, 'ram_gb': 16},
    },
}
DEFAULT = 'default'


def level(name):
    return LEVELS[name or DEFAULT]


def suggest(gpu, vram_mib, display=None, ram=0):
    """(level id, why) for a graphics card (name, video memory), the display's (width, height) and the PC's memory
    in bytes, as far as they are known."""
    have, mem = vram_mib / 1024, ram / (1 << 30)
    pc = ', '.join(filter(None, [gpu, f'{have:.0f} GB of video memory' if vram_mib else '',
                                 f'{mem:.0f} GB of memory' if ram else '',
                                 f'a {display[0]}x{display[1]} display' if display else '']))
    if not vram_mib:
        return DEFAULT, 'no graphics card could be read, so this is the usual level'
    for name, lv in LEVELS.items():
        needs = lv.get('needs', {})
        if have + 0.5 >= needs.get('vram_gb', 0) and (not ram or mem + 1 >= needs.get('ram_gb', 0)):
            return name, pc
    return list(LEVELS)[-1], f'{pc}; this is the lightest level'
