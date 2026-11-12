@echo off
rem SPDX-License-Identifier: GPL-3.0-or-later
rem DekooNut setup. Without arguments it opens the window; "setup.bat --help" shows the terminal form.
rem It runs on a Python 3.14 of its own in toolchain\python, fetched once: nothing has to be installed first.
setlocal
cd /d "%~dp0"
set "PY=toolchain\python\tools\python.exe"
if exist "%PY%" goto run
echo Fetching Python 3.14 into toolchain\python (once) ...
if not exist toolchain mkdir toolchain
if not exist toolchain\python mkdir toolchain\python
curl.exe -sSfL -o toolchain\python.zip https://www.nuget.org/api/v2/package/python/3.14.10 || goto failed
tar -xf toolchain\python.zip -C toolchain\python || goto failed
del toolchain\python.zip
if not exist "%PY%" goto failed
:run
"%PY%" -m setup %*
exit /b %ERRORLEVEL%
:failed
echo Python could not be fetched. Check the internet connection, or install Python 3.14 and run: python -m setup
exit /b 1
