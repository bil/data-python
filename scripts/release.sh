#!/usr/bin/env bash
set -e
if [[ -n "$(git status --porcelain)" ]] then
    echo "Nonempty status"
    exit 1
fi
if [[ ! -f .env ]] then
    echo "Missing .env"
    exit 1
fi
# push commits
git push origin HEAD:main
# tag commit
tag="v$(uv version --short)"
git tag "$tag"
# push tag
git push github $tag
# Expect PYPI_TOKEN in a .env
source .env
uv build
uv publish --token $PYPI_TOKEN
