@echo off
REM Re-add all staged Python files modified by Black
for /f "tokens=*" %%i in ('git diff --name-only --cached -- "*.py"') do git add "%%i"