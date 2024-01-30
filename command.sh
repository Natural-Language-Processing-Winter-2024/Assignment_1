#!/bin/bash

git add .
git commit -m "$1"
git push
git checkout main
git merge working
git push
git checkout working

echo "Press any key to continue..."
read -n 1 -s
