@echo off
rem SPDX-License-Identifier: GPL-3.0-or-later
rem DekooNut setup. Without arguments it opens the window; "setup.bat --help" shows the terminal form.
cd /d "%~dp0"
where py >nul 2>nul && (py -3 -m setup %* & exit /b)
where python >nul 2>nul && (python -m setup %* & exit /b)
echo DekooNut setup needs Python 3.14: winget install -e --id Python.Python.3.14
exit /b 1
