mlir-aie-dir := shell("python3 -c 'from aie.utils.config import root_path; print(root_path())'")
warning-flags := "-Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body -Wno-missing-template-arg-list-after-template-kw"
peano-flags := f"-O2 -std=c++20 --target=aie2p-none-unknown-elf -DNDEBUG {{warning-flags}}"

# show compute tile assembly
show-asm: xclbin
    ${PEANO_INSTALL_DIR}/bin/llvm-objdump -C -z -d build/{{project}}.mlir.prj/main_core_0_2.elf

# build C++ kernel
kernel: _build_dir
    ${PEANO_INSTALL_DIR}/bin/clang++ {{peano-flags}} -I {{mlir-aie-dir}}/include \
        -c {{project}}.cc -o build/{{project}}.o

_build_dir:
    mkdir -p build

_aiecc:
   aiecc --tmpdir build/{{project}}.mlir.prj \
        --aie-generate-xclbin --aie-generate-npu-insts \
        --no-xchesscc --no-xbridge \
        --xclbin-name=build/{{project}}.xclbin --npu-insts-name=build/{{project}}_insts.bin \
        build/{{project}}.o build/{{project}}.mlir

_process_trace:
    python3 {{mlir-aie-dir}}/python/aie/utils/trace/parse.py --input build/trace.txt \
        --mlir build/{{project}}.mlir.prj/input_with_addresses.mlir --output {{project}}_trace.json
    python3 {{mlir-aie-dir}}/python/aie/utils/trace/get_trace_summary.py --input {{project}}_trace.json

# lint the code
lint *args:
    uvx ruff check {{args}}

# format the code
format:
    find . -iname '*.h' -o -iname '*.cpp' -o -iname '*.cc' -o -iname '*.c' | xargs clang-format -i
    uvx ruff format

# clean build outputs
clean:
    rm -rf build
