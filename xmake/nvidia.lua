local nvidia_cuda_arch = get_config("cuda-arch") or "sm_80"

target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cxx17", "cuda")
    set_warnings("all", "error")
    set_toolset("cu", "nvcc")
    set_policy("build.cuda.devlink", true)
    add_defines("ENABLE_NVIDIA_API")

    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("-Xcompiler=-fPIC", "-arch=" .. nvidia_cuda_arch,
                    "--default-stream=per-thread")
        add_culdflags("-Xcompiler=-fPIC", "-arch=" .. nvidia_cuda_arch)
    end

    add_files("../src/device/nvidia/*.cu")
    add_links("cublas", "cudart")

    on_install(function (target) end)
target_end()

if has_config("flashinfer") then
target("llaisys-flashinfer-nvidia")
    set_kind("static")
    set_languages("cxx17", "cuda")
    set_toolset("cu", "nvcc")
    set_policy("build.cuda.devlink", true)
    add_defines("ENABLE_NVIDIA_API", "ENABLE_FLASHINFER")

    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("-Xcompiler=-fPIC", "-arch=" .. nvidia_cuda_arch,
                    "--default-stream=per-thread", "-w")
        add_culdflags("-Xcompiler=-fPIC", "-arch=" .. nvidia_cuda_arch)
    end

    add_files("../src/ops/self_attention/nvidia/flashinfer_adapter.cu")
    add_links("cudart")

    on_install(function (target) end)
target_end()
end

target("llaisys-ops-nvidia")
    set_kind("static")
    set_languages("cxx17", "cuda")
    add_deps("llaisys-tensor")
    set_warnings("all", "error")
    set_toolset("cu", "nvcc")
    set_policy("build.cuda.devlink", true)
    add_defines("ENABLE_NVIDIA_API")
    if has_config("flashinfer") then
        add_deps("llaisys-flashinfer-nvidia")
    end

    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("-Xcompiler=-fPIC", "-arch=" .. nvidia_cuda_arch,
                    "--default-stream=per-thread")
        add_culdflags("-Xcompiler=-fPIC", "-arch=" .. nvidia_cuda_arch)
    end

    add_files("../src/ops/*/nvidia/*.cu")
    remove_files("../src/ops/self_attention/nvidia/flashinfer_adapter.cu")
    add_links("cublas", "cudart")

    on_install(function (target) end)
target_end()
