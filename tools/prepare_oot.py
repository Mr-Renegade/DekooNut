"""Prepare every shader of Ocarina of Time before play: one command, safe to stop and run again.

    python3.14 tools/prepare_oot.py build/oot111 --romfs build/romfs-oot111 --xlate build/xlate/oot_xlate

tools/prepare_oot.sh runs it under a memory cap.

Steps; each leaves its result next to the build's `oot` and is skipped when that result is there and current:

  census   tools/shader_census.py nvn2_inventory: every NVN 2 stage of the RomFS into seed-nvn2.bin.
  lower    `oot --lower-shaders`: each vertex, fragment and compute stage through runtime/nvn2_lower.cpp, as program
           creation does, so a stage has the code hash it has in play: seed-programs.bin, seed-lowered.bin. No device.
  plan     seed-answers.bin: which variants to translate (below).
  build    `oot --prepare-shaders`: translates each variant into shader-cache/, then builds each compute pipeline and
           stage library into the driver's disk cache (vk_pipeline.c lb_warm_prepared), skipping what shader-built.log
           says is done. No game runs.

The variants. A translation depends on the answers inside the shader's use mask (gfx_shader.c, "answers met in play"),
which only the translator can tell; so plan and build run in two rounds: the first asks for each shader's default
variant, the second reads the reflection (.res.json) and asks again where the shader itself shows better answers.
  vertex   attributes 6 and 7 are the skinning indices, uint; the rest float.
  fragment float outputs.
  compute  the texture types the shader's own texture instructions name; image formats are in no shader, so those
           play uses most, --image-tries guesses each.
Variants recorded in play (this build's shader-answers.bin, or --answers DIR) come first: they are exact.

Order: play's variants, then per stage kind (--stages) the stages, Product.100 last.

Never built: stages with tensor instructions (emulated HMMA: megabytes of SPIR-V, and the driver takes minutes and
more memory than a game machine has on them; runtime skips those dispatches too).

Game-derived output stays in the build directory. Needs Python 3.14 (compression.zstd), standard library only.
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import shader_census  # noqa: E402

WIN = os.name == 'nt'
EXE = 'oot.exe' if WIN else 'oot'
if not WIN:
    import fcntl
# one game or GPU job at a time across worktrees (Linux; Windows takes no lock)
LOCK = str(Path(tempfile.gettempdir()) / 'oot-game.lock') if WIN else '/tmp/oot-game.lock'
VS, FS, CS = 0, 4, 5
PRG_HEAD = 12 + 32 + 0x880
ANS_PACKED, ANS_REC = 148, 4 + 32 + 32 + 148 + 1          # gfx_shader.c ANS_REC, version 2
LOWERED_REC = 8 + 16 + 32
UINT = 2
TEX_2D, TEX_2DARRAY, TEX_3D, TEX_CUBE = 4, 5, 8, 9        # gfx_shader.c tex_names
PRIORITY_RECORDED, PRIORITY_SEEN, PRIORITY_REST = 0, 1, 2


def log(text):
    print(f'[{time.strftime("%H:%M:%S")}] {text}', flush=True)


def census(build, romfs, workers):
    out = build / 'seed-nvn2.bin'
    if out.is_file() and out.stat().st_mtime >= (romfs / 'index.json').stat().st_mtime:
        log(f'census: {out.name} is there')
        return
    t0 = time.monotonic()
    shaders = shader_census.nvn2_inventory(romfs, workers)
    tmp = out.with_suffix('.tmp')
    with tmp.open('wb') as f:
        f.write(b'OOTNVN2S' + struct.pack('<II', 1, len(shaders)))
        for key, (stage, ctl, code, _) in sorted(shaders.items()):
            f.write(struct.pack('<IIII', shader_census.NVN2_STAGE_ID[stage], len(ctl), len(code), 0) +
                    bytes.fromhex(key) + ctl + code)
    tmp.replace(out)
    with (build / 'seed-sources.tsv').open('w', encoding='utf-8') as f:
        for key, (stage, _, _, source) in sorted(shaders.items()):
            f.write(f'{key}\t{stage}\t{source}\n')
    by = Counter(s[0] for s in shaders.values())
    log(f'census: {len(shaders)} stages {dict(by)} in {time.monotonic() - t0:.1f} s')


def run_env(xlate):
    env = dict(os.environ)
    if xlate:
        env['OOT_XLATE'] = str(xlate)
    return env


def lower(build, shards, xlate):
    src, prg, idx = build / 'seed-nvn2.bin', build / 'seed-programs.bin', build / 'seed-lowered.bin'
    exe = build / EXE
    if idx.is_file() and prg.is_file() and idx.stat().st_mtime >= max(src.stat().st_mtime, exe.stat().st_mtime):
        log('lower: seed-programs.bin is current')
        return
    t0 = time.monotonic()
    env = run_env(xlate)
    runs = []
    for i in range(shards):
        with open(build / f'lower.{i}.log', 'wb') as out:
            runs.append(subprocess.Popen([str(exe), '--lower-shaders'], cwd=build, stdout=out, stderr=subprocess.STDOUT,
                                         env=dict(env, OOT_LOWER_SHARD=f'{i}/{shards}')))
    if any(r.wait() for r in runs):
        sys.exit('lower: a shard failed, see lower.*.log')
    with prg.with_suffix('.tmp').open('wb') as f, idx.with_suffix('.tmp').open('wb') as g:
        f.write(b'OOT_SPRG' + struct.pack('<II', 1, 0))
        for i in range(shards):
            for part, to in ((f'seed-programs.{i}.bin', f), (f'seed-lowered.{i}.bin', g)):
                with open(build / part, 'rb') as p:
                    while chunk := p.read(1 << 24):
                        to.write(chunk)
                os.unlink(build / part)
    prg.with_suffix('.tmp').replace(prg)
    idx.with_suffix('.tmp').replace(idx)
    log(f'lower: {idx.stat().st_size // LOWERED_REC} stages, {prg.stat().st_size >> 20} MiB in {time.monotonic() - t0:.1f} s')


def read_lowered(build):
    """code hash -> dict(key, stage, flags, length); stages that did not lower are counted, not returned."""
    data = (build / 'seed-lowered.bin').read_bytes()
    rows, failed = {}, Counter()
    for o in range(0, len(data), LOWERED_REC):
        stage, flags, length = struct.unpack_from('<III', data, o + 8)
        if not flags & 1:
            failed[stage] += 1
            continue
        rows[data[o + 24:o + 56]] = dict(key=data[o:o + 8].hex(), stage=stage, flags=flags, length=length)
    return rows, failed


def walk_programs(path):
    """(offset of the record, stage, length, c1 length, code hash) of a OOT_SPRG file."""
    with open(path, 'rb') as f:
        if f.read(8) != b'OOT_SPRG':
            raise ValueError(f'{path}: not a program file')
        o, size = 16, os.fstat(f.fileno()).st_size
        while o + PRG_HEAD <= size:
            f.seek(o)
            head = f.read(44)
            stage, length, c1 = struct.unpack_from('<III', head)
            yield o, stage, length, c1, head[12:44]
            o += PRG_HEAD + length + c1


def read_answers(path):
    """stage, code hash, input id, packed answers of a OOT_SANS file (version 2)."""
    b = path.read_bytes()
    if b[:8] != b'OOT_SANS' or struct.unpack_from('<II', b, 8) != (2, ANS_REC):
        raise ValueError(f'{path}: not a version 2 answers file')
    for o in range(16, len(b) - ANS_REC + 1, ANS_REC):
        yield struct.unpack_from('<I', b, o)[0], b[o + 4:o + 36], b[o + 36:o + 68], b[o + 68:o + 68 + ANS_PACKED]


def pack(tex=None, img=None, fmt=None, attr=None, fout=None, local=(0, 0, 0)):
    a = bytearray(ANS_PACKED)
    for base, table in ((0, tex), (32, img), (64, fmt), (96, attr), (128, fout)):
        for slot, value in (table or {}).items():
            a[base + slot] = value
    struct.pack_into('<HHH', a, 136, *local)
    return bytes(a)


def local_of(control):
    return tuple(v or 1 for v in struct.unpack_from('<III', control, 0x748))


DEFAULT_TEX = {slot: TEX_2D for slot in range(32)}


def default_answers(stage, control):
    if stage == VS:
        return pack(attr={6: UINT, 7: UINT})
    if stage == CS:
        return pack(tex=DEFAULT_TEX, local=local_of(control))
    return pack(tex=DEFAULT_TEX)


def sampler_code(sampler):
    """The translator's SamplerType -> gfx_shader.c tex_names index."""
    base, array, ms = sampler & 0xff, sampler & 0x100, sampler & 0x200
    if base == 3:
        return (7 if array else 6) if ms else (TEX_2DARRAY if array else TEX_2D)
    return {1: 2 if array else 1, 2: 3, 4: TEX_3D, 5: 10 if array else TEX_CUBE}.get(base, TEX_2D)


def slot_of(res, handle, images):
    """The slot of a compute resource's packed-bank handle (gfx_shader.c parse_res, nvn2_c0.h)."""
    c0, roots = res.get('c0'), res.get('c0_roots') or []
    if not c0:
        return -1
    at = 0x10 + sum((w + 3 & ~3) * 4 for w in roots) + 16 * (c0[1] + c0[2]) + (8 * c0[3] if images else 0)
    bytes_ = (handle & 0x3fff) * 4
    count = c0[0] if images else c0[3]
    return (bytes_ - at) // 8 if at <= bytes_ < at + 8 * count and (bytes_ - at) % 8 == 0 else -1


def format_order(recorded):
    """Image formats by how often play bound them, per image dimension: gfx_shader.c fmt codes."""
    by = defaultdict(Counter)
    for tex, img, fmt in recorded:
        for slot in range(32):
            if img[slot] and fmt[slot]:
                by[img[slot]][fmt[slot]] += 1
    return {k: [f for f, _ in v.most_common()] for k, v in by.items()}


def compute_variants(res, local, formats, tries):
    """The answers a compute shader's own reflection points at, best guess first. A texture's dimension is in its
    instruction, so a type query (TXQ) has its answer in the resource of that slot. An image's format is nowhere in the
    shader: `tries` guesses, every image of the shader taking its dimension's n-th most played format."""
    tex, img = dict(DEFAULT_TEX), {}
    for t in res['textures']:
        slot = slot_of(res, t['handle'], False)
        if 0 <= slot < 32:
            tex[slot] = sampler_code(t['type'])
    images = [(slot_of(res, i['handle'], True), sampler_code(i['type'])) for i in res['images']]
    images = [(s, t) for s, t in images if 0 <= s < 32]
    if not images:
        return [pack(tex=tex, local=local)]
    out = []
    for n in range(tries * 3 if len(images) == 1 else tries):   # one image: guesses are cheap and it is the commonest
        fmt = {}
        for slot, t in images:
            img[slot] = t
            order = formats.get(t) or formats.get(TEX_2D) or [29]
            fmt[slot] = order[min(n, len(order) - 1)]
        out.append(pack(tex=tex, img=img, fmt=fmt, local=local))
    return out


def plan(build, recorded_dir, stages, no_recorded, image_tries):
    """Writes seed-answers.bin: variants recorded in play, then per stage kind in the order asked for: the
    stages, then the other Product version. The reflection of an earlier round refines
    compute variants. Returns the number of variants."""
    rows, failed = read_lowered(build)
    source = {}
    for line in (build / 'seed-sources.tsv').read_text(encoding='utf-8').splitlines():
        key, _, name = line.split('\t')
        source[key] = name
    controls = {}
    with open(build / 'seed-programs.bin', 'rb') as f:
        for o, stage, length, c1, h in walk_programs(build / 'seed-programs.bin'):
            if stage == CS:
                f.seek(o + 44)
                controls[h] = f.read(0x880)
    played, played_formats = [], []
    answers_file = (recorded_dir or build) / 'shader-answers.bin'
    if answers_file.is_file():
        for stage, h, _, packed in read_answers(answers_file):
            played_formats.append((packed[:32], packed[32:64], packed[64:96]))
            if h in rows and not rows[h]['flags'] & 2 and not no_recorded:
                played.append((stage, h, packed))
    formats = format_order(played_formats)
    reflected = reflection_of(build)
    recs, have = [], set()

    def add(rank, stage, h, answers):
        if (h, answers) not in have:
            have.add((h, answers))
            recs.append((rank, struct.pack('<I', stage) + h + bytes(32) + answers + bytes([min(rank, 2)])))

    for stage, h, packed in played:
        add(0, stage, h, packed)
    skipped, refined = Counter(), 0
    for h, row in rows.items():
        stage = row['stage']
        if stage not in stages:
            skipped['stage not asked for'] += 1
            continue
        if row['flags'] & 2:
            skipped['tensor'] += 1
            continue
        rank = 1 + 10 * stages.index(stage) + (1 if '.Product.100.' not in source.get(row['key'], '') else 2)
        control = controls.get(h, bytes(0x880))
        if stage == CS and h in reflected:
            for answers in compute_variants(reflected[h], local_of(control), formats, image_tries):
                add(rank, stage, h, answers)
                refined += 1
        else:
            add(rank, stage, h, default_answers(stage, control))
    recs.sort(key=lambda r: r[0])
    tmp = build / 'seed-answers.tmp'
    with open(tmp, 'wb') as f:
        f.write(b'OOT_SANS' + struct.pack('<II', 2, ANS_REC))
        for _, rec in recs:
            f.write(rec)
    tmp.replace(build / 'seed-answers.bin')
    (build / 'seed-keys.bin').unlink(missing_ok=True)
    log(f'plan: {len(recs)} variants ({len(played)} from play, {refined} from reflection) of {len(rows)} lowered stages; '
        f'lowering failed {dict(failed)}; left out {dict(skipped)}')
    return len(recs)


def reflection_of(build):
    """code hash -> the .res.json of the last translation of that shader (seed-answers.bin with seed-keys.bin)."""
    keys, answers = build / 'seed-keys.bin', build / 'seed-answers.bin'
    if not keys.is_file() or not answers.is_file():
        return {}
    k = keys.read_bytes()
    out = {}
    for i, (stage, h, _, _) in enumerate(read_answers(answers)):
        if stage == CS and h not in out and any(k[i * 32:i * 32 + 32]):
            p = build / 'shader-cache' / f'{k[i * 32:i * 32 + 32].hex()}.res.json'
            if p.is_file():
                out[h] = json.loads(p.read_text(encoding='utf-8'))
    return out


def game_running():
    if WIN:
        out = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq oot.exe', '/NH'], capture_output=True, text=True).stdout
        return 'oot.exe' in out.lower()
    return subprocess.run(['pgrep', '-x', 'oot'], capture_output=True).returncode == 0


def oot_prepare(build, threads, xlate, what, slice_s=None, lock=True):
    """`oot --prepare-shaders` until it finishes: in slices, each holding the game lock, so a test or a player waits one
    slice at most; again if the process dies, the journal going on after the module that killed it. Translating is CPU
    work in a separate process and takes no lock."""
    env = run_env(xlate)
    if not WIN:
        env.setdefault('WAYLAND_DISPLAY', 'wayland-0')
        env.setdefault('XDG_RUNTIME_DIR', f'/run/user/{os.getuid()}')
    env.update(what)
    env.setdefault('OOT_PREPARE_BIG_BUILDS', '6')
    if slice_s:
        env['OOT_PREPARE_SLICE_S'] = str(slice_s)
    deaths = 0
    while True:
        with open(LOCK, 'w') as held:
            if lock and not WIN:
                fcntl.flock(held, fcntl.LOCK_EX)
            if lock:
                while game_running():                   # a player's game has no lock: wait for it
                    time.sleep(10)
            n = threads or (6 if game_running() else 10)   # the PC is shared
            env.update(OOT_PREPARE_THREADS=str(n), OOT_PREWARM_XLATE=str(max(1, n // 2)))
            t0 = time.monotonic()
            with open(build / 'prepare.log', 'ab') as out:
                r = subprocess.run([str(build / EXE), '--prepare-shaders'], cwd=build, stdout=out,
                                   stderr=subprocess.STDOUT, env=env)
        with open(build / 'prepare.log', 'rb') as f:
            tail = [l.decode('utf-8', 'replace').rstrip() for l in f if l.startswith(b'prepare-shaders:')][-3:]
        log(f'oot --prepare-shaders: rc={r.returncode}, {time.monotonic() - t0:.0f} s; ' + ' | '.join(tail[-2:]))
        if r.returncode:
            deaths += 1
            if deaths > 8:
                sys.exit('the preparation keeps dying: see prepare.log')
            continue
        if not any('slice over' in line for line in tail):
            return
        time.sleep(3)                                   # let whoever waits for the lock take it


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('build', type=Path, help='build directory holding oot')
    ap.add_argument('--romfs', type=Path, required=True, help='directory with romfs.bin and index.json')
    ap.add_argument('--xlate', type=Path, help='oot_xlate (OOT_XLATE)')
    ap.add_argument('--stages', default='cs,vs,ps', help='stages to prepare, in this order')
    ap.add_argument('--threads', type=int, help='CPU threads (default 10, 6 beside a game; census and lowering at most 6)')
    ap.add_argument('--no-recorded', action='store_true', help='leave out recorded variants (to measure the defaults)')
    ap.add_argument('--answers', type=Path, help='directory of a shader-answers.bin (default the build directory)')
    ap.add_argument('--image-tries', type=int, default=3, help='format guesses per compute shader with images')
    ap.add_argument('--slice', type=int, default=600, help='seconds of building per hold of the game lock; 0 no slices')
    ap.add_argument('--no-lock', action='store_true', help='do not take /tmp/oot-game.lock: the build is CPU work in the '
                    "driver's compiler (GPU at 0%%), so it can run beside a game; only a timed comparison needs the lock")
    ap.add_argument('--until', choices=('census', 'lower', 'plan', 'translate', 'build'), default='build')
    a = ap.parse_args()
    build = a.build.resolve()
    t0 = time.monotonic()
    workers = min(a.threads or 6, 6)
    census(build, a.romfs.resolve(), workers)
    if a.until == 'census':
        return 0
    lower(build, workers, a.xlate)
    if a.until == 'lower':
        return 0
    stages = [{'vs': VS, 'ps': FS, 'cs': CS}[s] for s in a.stages.split(',')]
    answers = a.answers.resolve() if a.answers else None
    # Round 1 translates what has no reflection yet; round 2 uses the reflection. Neither compiles for the GPU.
    plan(build, answers, stages, a.no_recorded, a.image_tries)
    if a.until == 'plan':
        return 0
    translate_only = dict(OOT_PREPARE_TRANSLATE_ONLY='1')
    oot_prepare(build, a.threads, a.xlate, translate_only, lock=False)
    plan(build, answers, stages, a.no_recorded, a.image_tries)
    oot_prepare(build, a.threads, a.xlate, translate_only, lock=False)
    if a.until == 'translate':
        return 0
    oot_prepare(build, a.threads, a.xlate, dict(OOT_PREPARE_BUILD_ONLY='1'), a.slice, lock=not a.no_lock)
    log(f'done in {time.monotonic() - t0:.0f} s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
