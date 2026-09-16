#!/usr/bin/env sh
set -eu

repo="${FPL_STRATEGY_REPO:-bsovs/fpl-strategy-mcp}"
version="${FPL_STRATEGY_VERSION:-latest}"
install_dir="${FPL_STRATEGY_INSTALL_DIR:-$HOME/.local/bin}"
clients="${FPL_STRATEGY_CLIENTS:-none}"

usage() {
  cat <<'EOF'
Install the FPL Strategy MCP binary.

Usage:
  install.sh [--clients claude,codex|all] [--version VERSION]

The client option also registers the server with Claude Desktop/Code and/or
the Codex CLI. Existing Claude JSON is backed up before it is changed.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --clients)
      [ "$#" -ge 2 ] || { echo "--clients requires a value" >&2; exit 2; }
      clients="$2"
      shift 2
      ;;
    --version)
      [ "$#" -ge 2 ] || { echo "--version requires a value" >&2; exit 2; }
      version="$2"
      shift 2
      ;;
    --install-dir)
      [ "$#" -ge 2 ] || { echo "--install-dir requires a value" >&2; exit 2; }
      install_dir="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

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

if [ "$clients" != "none" ] && [ -n "$clients" ]; then
  "$install_dir/fpl-strategy-mcp" setup --clients "$clients"
fi

"$install_dir/fpl-strategy-mcp" status

case ":${PATH}:" in
  *":$install_dir:"*) ;;
  *) echo "Add $install_dir to PATH if it is not already available." ;;
esac
