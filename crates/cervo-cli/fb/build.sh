set -euxo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON_DIR="../python/messages"
RUST_DIR="../src/commands/serve/"
mkdir -p "$PYTHON_DIR"
mkdir -p "$RUST_DIR"

capnp --verbose compile -o"rust:$RUST_DIR" request.capnp
cp request.capnp ../python

pipx install --python=python3.11 'capnp-stub-generator @ git+https://github.com/commaai/capnp-stub-generator'
capnp-stub-generator -p '*.capnp' -c "$PYTHON_DIR/*_capnp.py" "$PYTHON_DIR/*_capnp.pyi"
mv *.{py,pyi} ../python

find "../src/commands/serve/" -type f -exec sed -i 's/crate::request_capnp/crate::commands::serve::request_capnp/g' {} \;
