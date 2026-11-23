#!/usr/bin/env bash
# Prepare one Linux x86_64 engine and its pinned weights.
set -euo pipefail
backend=${1:-nvfp4}
if [[ $# -gt 1 || ( $backend != nvfp4 && $backend != q4 ) ]]; then
    echo "Usage: bash tools/prepare_realtime.sh [nvfp4|q4]" >&2
    exit 2
fi
runtime="$(cd "$(dirname "$0")/.." && pwd)/.runtime"
mkdir -p "$runtime"
cd "$runtime"
exec 9>prepare-realtime-linux.lock
flock -n 9 || { echo "Another preparation is in progress." >&2; exit 1; }

fetch() {
    local url=$1 file=$2 digest=$3
    if [[ -f "$file" ]]; then
        printf '%s  %s\n' "$digest" "$file" | sha256sum -c -
    else
        curl --fail --location --retry 3 --continue-at - --output "$file.partial" "$url"
        printf '%s  %s\n' "$digest" "$file.partial" | sha256sum -c -
        mv -- "$file.partial" "$file"
    fi
}

model_variable=JR_DRAMA3_${backend^^}_MODEL
model_path=${!model_variable:-}
if [[ -n $model_path ]]; then
    [[ $model_path = /* ]] || { echo "$model_variable must be an absolute GGUF path." >&2; exit 2; }
    mkdir -p -- "$(dirname "$model_path")"
fi

if [[ $backend == q4 ]]; then
    model=Drama-35B-A3B-v3-Q4_K.gguf
    repo=https://huggingface.co/sakasegawa/drama-3-GGUF/resolve/9a01354d04ab6eb521bc9cdfc56ab6824c474af2
    fetch "$repo/$model" "${model_path:-$model}" 2a57b14c9b208a2ff833ab7305b920ca8f154ad807004a52d96a86a90a55523e
    fetch https://github.com/nyosegawa/speech.cpp/releases/download/v0.11.2/speech-0.11.2-linux-x64-vulkan.zip \
        speechcpp-linux.zip 28e6841c650c8a4d7c5be06b8e34eb2f14d43f8b26d52fc73f24b504875dc12d
    unzip -oq speechcpp-linux.zip -d speechcpp
    chmod +x speechcpp/speech
    exit 0
fi

model=Drama3-V2-78e5ab7-NVFP4-256E.gguf
repo=https://huggingface.co/WonderChord/Drama3-V2-NVFP4-256E-GGUF/resolve/06cfd87d3aa87cdd37e816fac6da485b8f6cbdb4
fetch "$repo/$model" "${model_path:-$model}" 5577beb99be7bc0f4c8324daea90e2f3c8765d17842db41016560fb4820312b1
patch=drama3-nvfp4-audiocpp.patch
fetch "$repo/$patch" "$patch" bbb0742a2beb7c06a7592efaa5dca74f8bb1ba3f25da57ca9d882b874f082c09
commit=066ea9862f41f4d7bcf996d5f00e464d8d28ace5
if [[ ! -d audio.cpp ]]; then
    staging=$(mktemp -d audio.cpp-XXXXXX)
    git -C "$staging" init --quiet
    git -C "$staging" remote add origin https://github.com/0xShug0/audio.cpp.git
    git -C "$staging" fetch --depth=1 origin "$commit"
    git -C "$staging" checkout --detach FETCH_HEAD
    mv "$staging" audio.cpp
fi
cd audio.cpp
[[ $(git rev-parse HEAD) == "$commit" ]] || { echo "Expected audio.cpp $commit" >&2; exit 1; }
git -c submodule.external/audio.cpp-server-frontends.url=https://github.com/0xShug0/audio.cpp-server-frontends.git \
    submodule update --init --depth 1
if ! git apply --quiet --reverse --check "$runtime/$patch"; then
    git apply "$runtime/$patch"
fi
bash scripts/build_linux.sh --backend cuda --cuda-arch 120a --build-type Release \
    --build-dir build/linux-cuda-release --model-set custom --models fish_drama3 \
    --deployment-build --target audiocpp_server --jobs "${JOBS:-8}"
