#!/usr/bin/env sh
set -eu

repo="${FPL_STRATEGY_REPO:-bsovs/fpl-strategy-mcp}"
version="${FPL_STRATEGY_VERSION:-latest}"
install_dir="${FPL_STRATEGY_INSTALL_DIR:-$HOME/.local/bin}"

os="$(uname -s)"
arch="$(uname -m)"
case "$os:$arch" in
  Darwin:arm64|Darwin:aarch64) artifact="fpl-strategy-mcp-macos-arm64" ;;
  Darwin:x86_64) artifact="fpl-strategy-mcp-macos-x64" ;;
  Linux:x86_64|Linux:amd64) artifact="fpl-strategy-mcp-linux-x64" ;;
  *)
    echo "Unsupported platform: $os $arch" >&2
    exit 1
    ;;
esac

if [ "$version" = "latest" ]; then
  url="https://github.com/$repo/releases/latest/download/$artifact"
else
  release="${version#v}"
  url="https://github.com/$repo/releases/download/v$release/$artifact"
fi

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT INT TERM
echo "Downloading $artifact from $url"
curl --fail --location --silent --show-error "$url" -o "$tmp"
mkdir -p "$install_dir"
chmod 755 "$tmp"
mv "$tmp" "$install_dir/fpl-strategy-mcp"
echo "Installed $install_dir/fpl-strategy-mcp"

case ":${PATH}:" in
  *":$install_dir:"*) ;;
  *) echo "Add $install_dir to PATH if it is not already available." ;;
esac

