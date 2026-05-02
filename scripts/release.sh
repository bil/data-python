#!/usr/bin/env bash
if [[ -n "$(git status --porcelain)" ]] then
    echo "Nonempty status"
    exit 1
fi
# Run AFTER bumping version
git tag "v$(uv version --short)"
git push --tags
# Expect PYPI_TOKEN in a .env
[ ! -f .env ] || export $(grep -v '^#' .env | xargs)
uv build
uv publish --token $PYPI_TOKEN
