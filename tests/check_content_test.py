"""Content guard against synthetic files in isolated Git indexes; no game data.

Run: python tests/check_content_test.py
"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
import check_content as content  # noqa: E402

SAFE = b'Harmless synthetic fixture.\n'
GENERATED = b'\xff\x00' + content.GENERATED_MARKER.encode() + b'\x00\xfe'
KEY = b'synthetic_' + b'key = ' + b'12' * 16 + b'\n'


class ContentTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        self.env = dict(os.environ, GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
                        GIT_ATTR_NOSYSTEM='1', GIT_TERMINAL_PROMPT='0')
        for name in ('GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE', 'GIT_COMMON_DIR'):
            self.env.pop(name, None)
        self.git('init')
        self.git('config', 'core.autocrlf', 'false')

    def git(self, *args):
        return subprocess.run(['git', *args], cwd=self.repo, env=self.env, check=True,
                              capture_output=True).stdout

    def stage(self, name, data):
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self.git('add', '--', name)
        return path

    def check(self, *, staged=True):
        return subprocess.run([sys.executable, str(ROOT / 'tools' / 'check_content.py')]
                              + (['--staged'] if staged else []),
                              cwd=self.repo, env=self.env, capture_output=True, text=True)

    def assert_rejected(self, result, message):
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn(message, result.stdout)

    def test_forbidden_index_blob_is_rejected_with_safe_worktree(self):
        path = self.stage('fixture.txt', GENERATED)
        path.write_bytes(SAFE)
        self.assert_rejected(self.check(), 'generated from game code')

    def test_safe_index_blob_is_accepted_with_forbidden_worktree(self):
        path = self.stage('fixture.txt', SAFE)
        path.write_bytes(GENERATED)
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout, '')

    def test_missing_worktree_still_checks_index(self):
        path = self.stage('fixture.txt', GENERATED)
        path.unlink()
        self.assert_rejected(self.check(), 'generated from game code')

    def test_safe_index_blob_needs_no_worktree_file(self):
        path = self.stage('fixture.txt', SAFE)
        path.unlink()
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_binary_index_payload_keeps_key_line_check(self):
        path = self.stage('fixture.txt', b'\xff\x00\n' + KEY)
        path.write_bytes(SAFE)
        self.assert_rejected(self.check(), 'contains a key in prod.keys format')

    def test_size_limit_uses_index_size(self):
        path = self.stage('fixture.txt', b'x' * (content.MAX_BYTES + 1))
        path.write_bytes(SAFE)
        self.assert_rejected(self.check(), 'over the 1024 KiB limit')

    def test_large_worktree_does_not_reject_safe_index_blob(self):
        path = self.stage('fixture.txt', SAFE)
        path.write_bytes(b'x' * (content.MAX_BYTES + 1))
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_filename_checks_do_not_need_worktree_files(self):
        for name, message in [('nested/main', 'name of a Switch executable module'),
                              ('nested/module.NSO', 'blocked file type .NSO')]:
            with self.subTest(name=name):
                # A synthetic NSO header, not an extracted executable.
                path = self.stage(name, b'NSO0' + bytes(60))
                path.unlink()
                self.assert_rejected(self.check(), message)
                self.git('rm', '--cached', '--', name)

    def test_git_special_and_unicode_paths(self):
        name = 'nested [directory]/-caf\u00e9 notes.txt'
        path = self.stage(name, GENERATED)
        path.write_bytes(SAFE)
        self.assert_rejected(self.check(), 'generated from game code')

    def test_removed_index_entry_is_not_checked(self):
        self.stage('fixture.txt', GENERATED)
        self.git('rm', '--cached', '--', 'fixture.txt')
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_default_mode_still_checks_worktree_bytes(self):
        path = self.stage('fixture.txt', SAFE)
        path.write_bytes(GENERATED)
        self.assert_rejected(self.check(staged=False), 'generated from game code')

    def test_default_mode_accepts_safe_worktree_with_forbidden_index(self):
        path = self.stage('fixture.txt', GENERATED)
        path.write_bytes(SAFE)
        result = self.check(staged=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_default_mode_keeps_untracked_and_ignored_file_policy(self):
        (self.repo / '.gitignore').write_text('ignored.txt\n')
        (self.repo / 'ignored.txt').write_bytes(GENERATED)
        self.assertEqual(self.check(staged=False).returncode, 0)
        (self.repo / 'untracked.txt').write_bytes(GENERATED)
        self.assert_rejected(self.check(staged=False), 'generated from game code')

    def test_release_size_exception_preserves_content_checks(self):
        path = self.repo / 'tool.bin'
        path.write_bytes(GENERATED + b'x' * content.MAX_BYTES)
        self.assertEqual(content.problems_for(path, allow_large=True), ['generated from game code'])
        path.write_bytes(KEY + b'x' * content.MAX_BYTES)
        self.assertEqual(content.problems_for(path, allow_large=True), [])


if __name__ == '__main__':
    unittest.main()
