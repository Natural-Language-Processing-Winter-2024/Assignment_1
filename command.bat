@echo off

setlocal EnableDelayedExpansion

cd /d "%~dp0"

git add .
git commit -m "%~1"
git push
git checkout main
git merge working
git push
git checkout working

endlocal

pause