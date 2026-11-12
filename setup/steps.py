# SPDX-License-Identifier: GPL-3.0-or-later
"""The setup steps: each runs the tools of README.md "Build" with the install folder's paths.

Install folder (the game reads this layout beside `oot`):
    oot, oot_xlate (+ .id), oot_mesh_xlate, romfs-index.bin, oot.ini, dekoonut (the launcher), dekoonut.png
    exefs/, romfs/ (romfs.bin, index.json), dlss/, gamedata/<pack>/, profile/ (saves), logs/
    shader-cache/, mesh-cache/, nvcache/, the seed and pipeline cache files (made by the shader step and in play)
    oot-install.json, .setup/ (generated C, objects, setup logs: kept, so an update recompiles only what changed)
Tools that do not depend on the game (the translator, the mesh shader compiler, the decoder helpers) are built once
per checkout under build/setup/ and build/zbic/ and copied in.
"""
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import ROOT, VERSION, fetch, host, quality
from .runner import SetupError, Step, digest

GiB = 1 << 30
GAME = 'oot111'
EXPECTED = ROOT / 'tools' / 'data' / f'expected-{GAME}.json'
TOOLS = ROOT / 'build' / 'setup'
ROMFS_NEED = 27 * GiB     # the RomFS image
REST_NEED = 9 * GiB       # generated C, objects, the game, shader and pipeline caches
ICON = 'dekoonut.png'
WIN, EXE = host.WIN, host.EXE
OOT, XLATE, MESH = 'oot' + EXE, 'oot_xlate' + EXE, 'oot_mesh_xlate' + EXE
LAUNCHER = 'dekoonut.bat' if WIN else 'dekoonut'
MENU_ENTRY = (Path(os.environ.get('APPDATA', Path.home())) / 'Microsoft' / 'Windows' / 'Start Menu' / 'Programs' /
              'DekooNut.lnk') if WIN else Path.home() / '.local' / 'share' / 'applications' / 'dekoonut.desktop'
MENU_MARK = 'X-DekooNut-Setup=1'
AMOUNT = re.compile(r'(\d+)\s*(?:/|of)\s*(\d+)')


# ---- helpers

def run(ctx, cmd, env=None, cwd=ROOT, on_line=None, what=None):
    """Run a tool; its lines go to the log and to on_line(line). SetupError when it fails."""
    cmd = [str(c) for c in cmd]
    ctx.reporter.log('$ ' + ' '.join(cmd))
    # Below normal priority: the PC stays usable while setup compiles. A group of its own: a cancel ends the whole tree.
    kw = {'creationflags': subprocess.BELOW_NORMAL_PRIORITY_CLASS | subprocess.CREATE_NEW_PROCESS_GROUP} if WIN \
        else {'start_new_session': True, 'preexec_fn': _nice}
    proc = subprocess.Popen(cmd, cwd=cwd, env=env or host.tool_env(), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True, errors='replace', **kw)
    ctx.child = proc
    tail = []
    try:
        for line in proc.stdout:
            for part in line.rstrip('\n').split('\r'):
                if part.strip():
                    ctx.reporter.log(part)
                    tail = (tail + [part])[-12:]
                    if on_line:
                        on_line(part)
        code = proc.wait()
    except KeyboardInterrupt:
        stop(proc)
        raise
    finally:
        ctx.child = None
    if ctx.cancelled:
        raise KeyboardInterrupt
    if code:
        raise SetupError(f'{what or Path(cmd[1] if len(cmd) > 1 else cmd[0]).name} failed (exit code {code}).',
                         '\n'.join(tail[-8:]))


def _nice():
    os.nice(10)


def stop(proc):
    """End a tool and everything it started."""
    try:
        if WIN:
            subprocess.run(['taskkill', '/T', '/F', '/PID', str(proc.pid)], capture_output=True)
        else:
            os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if not WIN:
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    except ProcessLookupError:
        pass


def amounts(ctx, text):
    """An on_line that turns "n/m" or "n of m" in a tool's line into progress."""
    def on_line(line):
        m = AMOUNT.search(line)
        if m and int(m.group(2)):
            ctx.progress(text, int(m.group(1)), int(m.group(2)))
    return on_line


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def tree_digest(*paths):
    """Fingerprint of source files and folders (names and contents)."""
    h = hashlib.sha256()
    for base in paths:
        files = sorted(p for p in base.rglob('*') if p.is_file() and '__pycache__' not in p.parts) \
            if base.is_dir() else [base]
        for p in files:
            h.update(p.relative_to(ROOT).as_posix().encode() + b'\0' + hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()[:32]


def install(src, dst):
    """Copy a built file into the install folder; the old one stays until the new one is whole."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + '.setup-part')
    shutil.copy2(src, part)
    os.replace(part, dst)


def expected():
    return json.loads(EXPECTED.read_text())


def packed(romfs):
    """A RomFS folder that already is the image tools/pack_romfs.py writes."""
    return (romfs / 'romfs.bin').is_file() and (romfs / 'index.json').is_file()


def ini_get(text, key):
    """The value of an uncommented `key = value` line, or None."""
    for line in text.splitlines():
        name, eq, value = line.partition('=')
        if eq and name.strip().lower() == key.lower():
            return value.strip()
    return None


def ini_set(text, key, value):
    """Set key: its own line, else a new line under the commented default, else at the end."""
    lines = text.splitlines()
    new = f'{key} = {value}'
    for i, line in enumerate(lines):
        name, eq, _ = line.partition('=')
        if eq and name.strip().lower() == key.lower():
            lines[i] = new
            break
    else:
        for i, line in enumerate(lines):
            name, eq, _ = line.lstrip('#; ').partition('=')
            if line.lstrip().startswith(('#', ';')) and eq and name.strip().lower() == key.lower():
                lines.insert(i + 1, new)
                break
        else:
            lines.append(new)
    return '\n'.join(lines) + '\n'


def game_input(ctx):
    """('nup', file) or ('folder', dir with exefs/ and romfs/); SetupError when it is neither."""
    given = ctx.options.game or ctx.state.get('game')
    if not given:
        raise SetupError('No game file was given.', 'Choose your .nup file, or the folder that holds the exefs '
                         'and romfs folders of your own copy of the game.')
    p = Path(given).expanduser().resolve()
    if p.is_file():
        return 'nup', p
    if p.is_dir() and (p / 'exefs' / 'main').is_file() and (p / 'romfs').is_dir():
        return 'folder', p
    if p.is_dir():
        raise SetupError('That folder does not hold the game files.', f'{p}\nIt needs an exefs folder (main, '
                         'main.npdm, rtld, sdk, subsdk0) and a romfs folder.')
    raise SetupError('The game file was not found.', str(p))


def input_print(ctx):
    kind, p = game_input(ctx)
    if kind == 'nup':
        st = p.stat()
        return [kind, str(p), st.st_size, st.st_mtime_ns]
    exefs = sorted((f.name, f.stat().st_size, f.stat().st_mtime_ns) for f in (p / 'exefs').iterdir() if f.is_file())
    romfs = p / 'romfs'
    return [kind, str(p), exefs, len(os.listdir(romfs)), romfs.stat().st_mtime_ns]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(f'oot_tool_{name}', ROOT / 'tools' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def play_env(dest):
    """What the launcher sets besides the quality level: a driver shader cache of the game's own (Linux driver)."""
    if WIN:
        return {}
    return {'__GL_SHADER_DISK_CACHE': '1', '__GL_SHADER_DISK_CACHE_PATH': str(dest / 'nvcache'),
            '__GL_SHADER_DISK_CACHE_SIZE': str(8 * GiB), '__GL_SHADER_DISK_CACHE_SKIP_CLEANUP': '1'}


# ---- the steps

class Check(Step):
    id, title, always = 'check', 'Checking this PC', True

    def run(self, ctx):
        kind, src = game_input(ctx)
        if kind == 'nup':
            self.check_nup(src)
        dest = ctx.dest
        if any(src == d or src in d.parents for d in (dest,)):
            raise SetupError('The install folder is inside the game files folder.', 'Choose another folder.')
        if dest.is_dir() and not (dest / 'oot-install.json').is_file() and \
                any(n not in ('.setup',) for n in os.listdir(dest)):
            raise SetupError('The install folder is not empty.', f'{dest}\nChoose an empty or new folder, or one '
                             'that setup made before.')
        if not host.cpu_ok():
            raise SetupError('This CPU cannot run the game.', 'The game is built for CPUs with AVX2 '
                             '(x86-64-v3): Intel from 2013 (Haswell), AMD from 2015 (Excavator) or Ryzen.')
        missing = host.missing_packages()
        if missing:
            command = host.install_command(missing)
            raise SetupError('This PC lacks build tools that have to be installed first.',
                             'Missing: ' + ', '.join(what for what, _ in missing) +
                             ('\nRun this in a terminal, then start setup again:' if command else
                              '\nInstall them with the system\'s package manager, then start setup again.'),
                             command)
        need = (0 if (dest / 'romfs' / 'index.json').is_file() else ROMFS_NEED) + \
               (GiB if (dest / LAUNCHER).is_file() else REST_NEED)
        free = host.free_bytes(dest)
        if free < need:
            raise SetupError('Not enough free space for the install folder.',
                             f'{dest}\n{need / GiB:.0f} GB needed, {free / GiB:.0f} GB free.')
        cards = host.gpus()
        if not cards:
            ctx.reporter.warning('No NVIDIA graphics card was found (nvidia-smi is missing or reports none). '
                                 'The game needs one with a current driver.')
        elif max(c[1] for c in cards) < 7000:
            ctx.reporter.warning(f'{cards[0][0]} has {cards[0][1] / 1024:.0f} GB of video memory; the game is '
                                 'tested with 8 GB and more.')
        ram = host.memory_bytes()
        if ram and ram < 15 * GiB:
            ctx.reporter.warning(f'This PC has {ram / GiB:.0f} GB of memory; the build and the game are tested '
                                 'with 16 GB and more.')
        suggested = quality.suggest(cards[0][0] if cards else '', cards[0][1] if cards else 0, ram=ram)[0]
        ctx.state.update(version=VERSION, game=str(src), quality=ctx.options.quality or
                         ctx.state.get('quality') or suggested)
        if ctx.state['quality'] not in quality.LEVELS:
            raise SetupError(f'There is no quality level "{ctx.state["quality"]}".',
                             'Levels: ' + ', '.join(quality.LEVELS))

    def check_nup(self, path):
        nx = load_tool('extract_nup')
        want = int(json.loads(EXPECTED.read_text())['title_id'], 16)
        try:
            nup = nx.Nup(path)
            base, update = nx.pick(nup, 'base'), nx.pick(nup, 'update')
            nup.close()
        except nx.NupError as e:
            raise SetupError('That file cannot be used.', f'{path}\n{e}\nSetup needs a .nup container that holds '
                             'the game and its update 1.1.1.') from None
        if base.title_id != want:
            raise SetupError('That file is another game.', f'{path}\nIt holds title {base.title_id:016X}; '
                             f'setup builds {want:016X}.')
        del update


class Tools(Step):
    id, title = 'tools', 'Getting the build tools'

    def key(self, ctx):
        return digest(fetch.DLSS_COMMIT, fetch.DOTNET_CHANNEL, (ROOT / 'tools' / 'requirements.txt').read_text())

    def current(self, ctx, record):
        return fetch.all_ready()

    def run(self, ctx):
        fetch.venv314(ctx, run)
        if WIN:
            fetch.win_git(ctx)
            fetch.win_tools(ctx, run)
        fetch.venv312(ctx, run)
        fetch.dotnet(ctx, run)
        fetch.dlss(ctx, run)


class Icon(Step):
    id, title = 'icon', 'Taking the icon from your game'

    def key(self, ctx):
        return digest(input_print(ctx), tree_digest(ROOT / 'tools' / 'game_icon.py', ROOT / 'tools' / 'icon'))

    def current(self, ctx, record):
        if (ctx.dest / ICON).is_file():
            ctx.reporter.icon(ctx.dest / ICON)
            return True
        return False

    def run(self, ctx):
        kind, src = game_input(ctx)
        out = ctx.dest / ICON
        ctx.dest.mkdir(parents=True, exist_ok=True)
        tries = [['--nup', src, '--package', 'update'], ['--nup', src, '--package', 'base']] if kind == 'nup' \
            else [['--romfs' if packed(src / 'romfs') else '--dir', src / 'romfs']]
        for args in tries:
            try:
                ico = ['--ico', ctx.dest / 'dekoonut.ico'] if WIN else []
                run(ctx, [host.python314(), ROOT / 'tools' / 'game_icon.py', *args, '--out', out, *ico])
                ctx.reporter.icon(out)
                return {'from': 'game'}
            except SetupError as e:
                ctx.reporter.log(f'icon: {e.text} {e.detail}')
        # The game still builds without its picture.
        ctx.reporter.warning("The game's own icon could not be read; the menu entry gets the DekooNut picture.")
        shutil.copyfile(ROOT / 'assets' / 'icon.png', out)
        return {'from': 'repository'}


class Exefs(Step):
    id, title = 'exefs', 'Reading the game program'

    def key(self, ctx):
        return digest(input_print(ctx), expected()['exefs_sha256'],
                      tree_digest(ROOT / 'tools' / 'nso_plain.py', ROOT / 'tools' / 'zbic'))

    def current(self, ctx, record):
        return all((ctx.dest / 'exefs' / n).is_file() for n in expected()['exefs_sha256'])

    def run(self, ctx):
        kind, src = game_input(ctx)
        expected = json.loads(EXPECTED.read_text())
        if kind == 'nup':
            dump = ctx.work / 'dump'
            run(ctx, [host.python314(), ROOT / 'tools' / 'extract_nup.py', src, '--package', 'base',
                      '--overlay-update', '--only', 'exefs', '--out', dump], what='Reading the game file')
            dump = dump / 'exefs'
        else:
            dump = src / 'exefs'
        missing = [n for n in expected['exefs_sha256'] if not (dump / n).is_file()]
        if missing:
            raise SetupError('The game program is incomplete.', 'Missing in the exefs folder: ' + ', '.join(missing))
        run(ctx, [sys.executable, ROOT / 'tools' / 'build_zbic.py'], what='Building the program decoder')
        out = ctx.work / 'exefs-plain'
        shutil.rmtree(out, ignore_errors=True)
        run(ctx, [sys.executable, ROOT / 'tools' / 'nso_plain.py', dump, out], what='Converting the game program')
        wrong = [n for n, want in expected['exefs_sha256'].items() if file_sha256(out / n) != want]
        if wrong:
            raise SetupError('This is not update 1.1.1 of the game.',
                             f'These program files differ from version {expected["display_version"]}: ' +
                             ', '.join(wrong) + '\nSetup builds that version only: the game with its update '
                             '1.1.1 applied.')
        shutil.rmtree(ctx.dest / 'exefs', ignore_errors=True)
        os.replace(out, ctx.dest / 'exefs')
        shutil.rmtree(ctx.work / 'dump', ignore_errors=True)
        return {'sha256': expected['exefs_sha256']}


class Romfs(Step):
    id, title = 'romfs', 'Copying the game data'

    def key(self, ctx):
        return digest(input_print(ctx), tree_digest(ROOT / 'tools' / 'pack_romfs.py'))

    def current(self, ctx, record):
        folder = ctx.dest / 'romfs'
        return (folder / 'index.json').is_file() and (folder / 'romfs.bin').is_file() and \
            (folder / 'romfs.bin').stat().st_size == record.get('size')

    def run(self, ctx):
        kind, src = game_input(ctx)
        folder = ctx.dest / 'romfs'
        if kind == 'folder' and packed(src / 'romfs'):
            return self.copy_image(ctx, src / 'romfs', folder)
        source = ['--nup', src] if kind == 'nup' else ['--dir', src / 'romfs']

        def on_line(line):
            m = re.search(r'([\d.]+) of ([\d.]+) GiB', line)
            if m:
                ctx.progress(f'{m.group(1)} of {m.group(2)} GB', int(float(m.group(1)) * 100),
                             int(float(m.group(2)) * 100))
        run(ctx, [host.python314(), ROOT / 'tools' / 'pack_romfs.py', *source, '--out', folder], on_line=on_line,
            what='Copying the game data')
        head = json.loads((folder / 'index.json').read_text())
        want = expected().get('romfs_sha256')
        if want and head['romfs_sha256'] != want:
            ctx.reporter.warning('The game data differs from the copy the game is tested with (another version, '
                                 'or changed files). The build goes on.')
        return {'size': head['romfs_size'], 'sha256': head['romfs_sha256'], 'files': len(head['files'])}

    @staticmethod
    def copy_image(ctx, src, folder):
        """The input already is the packed image: copy it and check it against its own index."""
        head = json.loads((src / 'index.json').read_text())
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'index.json').unlink(missing_ok=True)
        h, done, total = hashlib.sha256(), 0, head['romfs_size']
        with open(src / 'romfs.bin', 'rb') as a, open(folder / 'romfs.bin', 'wb') as b:
            while block := a.read(8 << 20):
                b.write(block)
                h.update(block)
                done += len(block)
                ctx.progress(f'{done / GiB:.1f} of {total / GiB:.1f} GB', done >> 20, total >> 20)
        if done != total or h.hexdigest() != head['romfs_sha256']:
            raise SetupError('The game data image does not match its index.', f'{src}\nromfs.bin and index.json '
                             'belong together; make the image again or give the .nup file.')
        shutil.copyfile(src / 'index.json', folder / 'index.json')
        return {'size': total, 'sha256': head['romfs_sha256'], 'files': len(head['files'])}


class Generate(Step):
    id, title = 'generate', 'Recompiling the game code to C'

    def key(self, ctx):
        return digest(ctx.keys['exefs'], tree_digest(ROOT / 'recomp'))

    def current(self, ctx, record):
        return (ctx.work / 'generated' / GAME / 'layout.json').is_file()

    def run(self, ctx):
        out = ctx.work / 'generated' / GAME
        jobs = ['--jobs', ctx.options.jobs] if ctx.options.jobs else []
        run(ctx, [host.python312(), ROOT / 'recomp' / 'gen.py', '--exefs', ctx.dest / 'exefs', '--out', out,
                  '--patches', ROOT / 'recomp' / f'patches-{GAME}.txt', *jobs],
            on_line=amounts(ctx, 'Writing C'), what='The recompiler')


class Translator(Step):
    id, title = 'translator', 'Building the shader translator'

    def identity(self):
        return load_tool('build_xlate').identity()

    def key(self, ctx):
        return digest(self.identity())

    def current(self, ctx, record):
        ident = ctx.dest / 'oot_xlate.id'
        return (ctx.dest / XLATE).is_file() and ident.is_file() and ident.read_text().strip() == self.identity()

    def run(self, ctx):
        out = TOOLS / 'xlate'
        built = out / 'oot_xlate.id'
        if not ((out / XLATE).is_file() and built.is_file() and built.read_text().strip() == self.identity()):
            ctx.progress('This takes a few minutes', force=True)
            run(ctx, [sys.executable, ROOT / 'tools' / 'build_xlate.py', '--out', out])
        install(out / XLATE, ctx.dest / XLATE)
        install(built, ctx.dest / 'oot_xlate.id')
        return {'identity': self.identity()}


class Mesh(Step):
    id, title = 'mesh', 'Building the mesh shader compiler'

    def key(self, ctx):
        return digest(tree_digest(ROOT / 'tools' / 'mesh', ROOT / 'tools' / 'build_mesh.py'))

    def current(self, ctx, record):
        return (ctx.dest / MESH).is_file() and (ctx.dest / 'mesh-cache').is_dir()

    def run(self, ctx):
        out = TOOLS / 'mesh'
        stamp = out / '.setup-key'
        if not ((out / MESH).is_file() and stamp.is_file() and stamp.read_text() == ctx.keys['mesh']):
            toolchain = ['--toolchain', ROOT / 'toolchain'] if WIN else []
            run(ctx, [sys.executable, ROOT / 'tools' / 'build_mesh.py', '--out', out, *toolchain,
                      '--jobs', host.jobs(ctx.options.jobs)], on_line=amounts(ctx, 'Compiling'))
            stamp.write_text(ctx.keys['mesh'])
        install(out / MESH, ctx.dest / MESH)
        # The folder is the game's sign that the compiler beside it and this cache are its own.
        (ctx.dest / 'mesh-cache').mkdir(exist_ok=True)


class Game(Step):
    id, title = 'game', 'Compiling the game'

    def key(self, ctx):
        return digest(ctx.keys['generate'], ctx.keys['romfs'], fetch.DLSS_COMMIT, expected()['exefs_sha256'],
                      tree_digest(ROOT / 'runtime', ROOT / 'tools' / 'build_game.py',
                                  ROOT / 'tools' / 'romfs_index.py'))

    def current(self, ctx, record):
        return (ctx.dest / OOT).is_file() and (ctx.dest / 'romfs-index.bin').is_file() and \
            any((ctx.dest / 'dlss').glob('nvngx_dlss.dll' if WIN else 'libnvidia-ngx-dlss.so.*'))

    def run(self, ctx):
        out = ctx.work / 'build'
        jobs = ['--jobs', ctx.options.jobs] if ctx.options.jobs else []
        (ctx.dest / 'profile').mkdir(exist_ok=True)
        run(ctx, [host.python312(), ROOT / 'tools' / 'build_game.py', '--version', GAME, '--expected', EXPECTED,
                  '--exefs', ctx.dest / 'exefs', '--romfs', ctx.dest / 'romfs',
                  '--gen', ctx.work / 'generated' / GAME, '--out', out, '--profile', ctx.dest / 'profile',
                  '--full-build', *self.system_args(), *jobs],
            on_line=amounts(ctx, 'Compiling'), what='The game build')
        install(out / OOT, ctx.dest / OOT)
        install(out / 'romfs-index.bin', ctx.dest / 'romfs-index.bin')
        if WIN:
            for dll in out.glob('*.dll'):   # the libraries the build put beside the program
                install(dll, ctx.dest / dll.name)
        model = fetch.dlss_model()
        install(model, ctx.dest / 'dlss' / model.name)
        return {'sha256': file_sha256(ctx.dest / OOT)}

    @staticmethod
    def system_args():
        tc = ROOT / 'toolchain'
        win = ['--toolchain', tc / 'llvm-mingw', '--deps', tc / 'deps-mingw', '--ninja', tc / 'bin' / 'ninja.exe']
        return (win if WIN else []) + ['--dlss-sdk', fetch.DLSS_DIR]


def prepare_shaders(ctx, phase, pattern, text):
    """tools/prepare_oot.py for one of its two long phases, each a step of its own so that the window's progress
    never starts again in the middle of a step: phase is its option (--until translate, --build-only). pattern finds
    the game's own progress lines of that phase in prepare.log, the file the tool sends them to."""
    dest = ctx.dest
    threads = max(2, min(6, (os.cpu_count() or 4) - 2))
    cmd = [host.python314(), ROOT / 'tools' / 'prepare_oot.py', dest, '--romfs', dest / 'romfs',
           '--xlate', dest / XLATE, '--threads', threads, '--no-lock', '--slice', 0, *phase]
    ram = host.memory_bytes()
    if shutil.which('systemd-run') and ram:   # a memory cap: the driver's compiler can take a lot
        cap = max(6, min(14, ram * 7 // 10 // GiB))
        cmd = ['systemd-run', '--user', '--scope', '--quiet', '-p', f'MemoryMax={cap}G', '-p',
               'MemorySwapMax=2G', '-p', 'OOMPolicy=continue'] + cmd
    env = host.tool_env()
    env.update(play_env(dest))
    if not WIN:
        (dest / 'nvcache').mkdir(exist_ok=True)
    log = dest / 'prepare.log'
    done = threading.Event()

    def watch():
        pos = log.stat().st_size if log.is_file() else 0
        while not done.wait(1):
            if not log.is_file():
                continue
            with open(log, 'rb') as f:
                f.seek(pos)
                lines = f.read().decode(errors='replace').splitlines()
                pos = f.tell()
            for line in lines:
                m = re.search(pattern, line)
                if m:
                    ctx.progress(text, int(m.group(1)), int(m.group(2)))

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        run(ctx, cmd, env=env, on_line=lambda line: ctx.progress(line.split('] ', 1)[-1][:90]),
            what='The shader preparation')
    finally:
        done.set()
        watcher.join()


class Translate(Step):
    id, title = 'translate', 'Translating the shaders'

    def key(self, ctx):
        return digest(ctx.keys['game'], ctx.keys['translator'],
                      tree_digest(ROOT / 'tools' / 'prepare_oot.py', ROOT / 'tools' / 'shader_census.py'))

    def current(self, ctx, record):
        return (ctx.dest / 'seed-answers.bin').is_file() and (ctx.dest / 'shader-prepared.bin').is_file() and \
            (ctx.dest / 'shader-cache').is_dir()

    def run(self, ctx):
        prepare_shaders(ctx, ['--until', 'translate'], r'prepare: (\d+) of (\d+) pairs', 'Translating')


class Shaders(Step):
    id, title = 'shaders', 'Compiling the shaders for this graphics card'

    def key(self, ctx):
        return digest(ctx.keys['translate'], 'build')

    def current(self, ctx, record):
        return (ctx.dest / 'shader-built.log').is_file()

    def run(self, ctx):
        prepare_shaders(ctx, ['--build-only'], r'prepare-shaders: built (\d+) of (\d+)', 'Compiling')


class Gamedata(Step):
    id, title = 'gamedata', 'Applying the quality level'

    def key(self, ctx):
        level = quality.level(ctx.state.get('quality'))
        return digest(ctx.keys['romfs'], level['pack'], level['mods'], tree_digest(
            ROOT / 'tools' / 'gamedata', ROOT / 'tools' / 'build_gamedata.py'))

    def current(self, ctx, record):
        pack = quality.level(ctx.state.get('quality'))['pack']
        return not pack or (ctx.dest / 'gamedata' / pack / 'manifest.tsv').is_file()

    def run(self, ctx):
        level = quality.level(ctx.state.get('quality'))
        if level['pack']:
            run(ctx, [host.python314(), ROOT / 'tools' / 'build_gamedata.py', level['pack'], *level['mods'],
                      '--romfs', ctx.dest / 'romfs', '--out', ctx.dest / 'gamedata'], what='The game-data pack')


class Finish(Step):
    id, title, always = 'finish', 'Finishing', True

    def run(self, ctx):
        dest = ctx.dest
        (dest / 'profile').mkdir(exist_ok=True)
        if not (dest / 'oot.ini').is_file():
            shutil.copyfile(ROOT / 'oot.ini.example', dest / 'oot.ini')
        if not (dest / ICON).is_file():
            shutil.copyfile(ROOT / 'assets' / 'icon.png', dest / ICON)
        level = quality.level(ctx.state.get('quality'))
        self.ini(ctx, level.get('ini', {}))
        env = dict(level['env'])
        export_pack = level['pack'] and 'romfs_mods' not in level.get('ini', {})   # a level may name it in oot.ini
        if WIN:
            lines = ['@echo off', 'rem Starts DekooNut. Setup writes this file again on every run.', 'cd /d "%~dp0"']
            if export_pack:
                lines.append(f'set "OOT_ROMFS_MODS=%~dp0gamedata\\{level["pack"]}"')
            lines += [f'if not defined {k} set "{k}={v}"' for k, v in env.items()]
            lines += [f'start "" "%~dp0{OOT}" %*', '']
        else:
            lines = ['#!/bin/sh', '# Starts DekooNut. Setup writes this file again on every run.',
                     'here=$(dirname "$(readlink -f "$0")")', 'cd "$here" || exit 1']
            if export_pack:
                lines.append(f'export OOT_ROMFS_MODS="$here/gamedata/{level["pack"]}"')
            lines += [f'export {k}="${{{k}:-{v}}}"' for k, v in env.items()]
            cache = play_env(Path('$here'))
            lines += ['# A driver shader cache of the game\'s own: the shared one is small and fills up.',
                      'mkdir -p "$here/nvcache"'] + [f'export {k}="{v}"' for k, v in cache.items()]
            lines += [f'exec ./{OOT} "$@"', '']
        launcher = dest / LAUNCHER
        launcher.write_text(('\r\n' if WIN else '\n').join(lines), encoding='utf-8', newline='')
        launcher.chmod(0o755)
        if not ctx.options.no_menu_entry:
            (self.shortcut if WIN else self.menu_entry)(ctx, launcher)
        if getattr(ctx.options, 'desktop_entry', False):   # asked for: --desktop-entry, the window's tick box
            if WIN:
                self.shortcut(ctx, launcher, desktop=True)
            else:
                ctx.reporter.warning('A desktop shortcut is written on Windows only for now.')

    @staticmethod
    def ini(ctx, keys):
        """Write the level's settings into oot.ini, except those the player changed since setup wrote them."""
        path = ctx.dest / 'oot.ini'
        text = path.read_text(encoding='utf-8')
        wrote = ctx.state.setdefault('ini', {})
        for key, value in keys.items():
            now = ini_get(text, key)
            if now is None or now == wrote.get(key):
                text = ini_set(text, key, value)
                wrote[key] = value
        for key in [k for k in wrote if k not in keys]:   # a key of the level before: back to the game's default
            if ini_get(text, key) == wrote.pop(key):
                text = '\n'.join(l for l in text.splitlines() if not (
                    '=' in l and l.partition('=')[0].strip().lower() == key.lower())) + '\n'
        path.write_text(text, encoding='utf-8')

    @staticmethod
    def shortcut(ctx, launcher, desktop=False):
        """The Start menu entry, or with desktop the same shortcut on the desktop (wherever Windows keeps that folder:
        it may be redirected). Its icon: dekoonut.ico from the game's picture (the icon step), else the program's."""
        icon = ctx.dest / 'dekoonut.ico'
        script = (
            "$lnk = if ($env:LNK) { $env:LNK } else { Join-Path ([Environment]::GetFolderPath('Desktop')) 'DekooNut.lnk' }; "
            "$s = (New-Object -ComObject WScript.Shell).CreateShortcut($lnk); "
            "if ((Test-Path $lnk) -and $s.Description -ne $env:MARK) { exit 3 }; "
            "$s.TargetPath = $env:TARGET; $s.WorkingDirectory = $env:WORKDIR; $s.Description = $env:MARK; "
            "$s.WindowStyle = 7; $s.IconLocation = $env:ICON; $s.Save()")
        env = dict(os.environ, LNK='' if desktop else str(MENU_ENTRY), TARGET=str(launcher), WORKDIR=str(ctx.dest),
                   MARK='DekooNut (written by setup)', ICON=str(icon if icon.is_file() else ctx.dest / OOT))
        r = subprocess.run(['powershell', '-NoProfile', '-Command', script], env=env, capture_output=True, text=True)
        what, where = ('The desktop shortcut', 'DekooNut on the desktop') if desktop else ('The Start menu entry', MENU_ENTRY)
        if r.returncode == 3:
            ctx.reporter.warning(f'{where} exists and was not written by setup: it is left as it is.')
        elif r.returncode:
            ctx.reporter.warning(f'{what} could not be written: ' + (r.stderr.strip()[:200] or 'unknown'))

    @staticmethod
    def menu_entry(ctx, launcher):
        if MENU_ENTRY.is_file() and MENU_MARK not in MENU_ENTRY.read_text(errors='replace'):
            ctx.reporter.warning(f'{MENU_ENTRY} exists and was not written by setup: it is left as it is.')
            return
        MENU_ENTRY.parent.mkdir(parents=True, exist_ok=True)
        MENU_ENTRY.write_text('\n'.join([
            '[Desktop Entry]', 'Type=Application', 'Name=DekooNut',
            'Comment=The Legend of Zelda: Ocarina of Time, built from your own copy',
            f'Exec="{launcher}"', f'Icon={ctx.dest / ICON}', 'Terminal=false', 'Categories=Game;',
            'StartupWMClass=dekoonut', MENU_MARK, '']), encoding='utf-8')


def all_steps():
    return [Check(), Tools(), Icon(), Exefs(), Romfs(), Generate(), Translator(), Mesh(), Game(), Translate(),
            Shaders(), Gamedata(), Finish()]
