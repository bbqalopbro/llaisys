target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cxx17", "cuda")
    set_warnings("all", "error")
    set_toolset("cu", "nvcc")
    set_policy("build.cuda.devlink", true)
    add_defines("ENABLE_NVIDIA_API")

    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("-Xcompiler=-fPIC", "-arch=sm_80", "--default-stream=per-thread")
        add_culdflags("-Xcompiler=-fPIC")
    end

    add_files("../src/device/nvidia/*.cu")
    add_links("cublas", "cudart")

    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    set_languages("cxx17", "cuda")
    add_deps("llaisys-tensor")
    set_warnings("all", "error")
    set_toolset("cu", "nvcc")
    set_policy("build.cuda.devlink", true)
    add_defines("ENABLE_NVIDIA_API")

    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("-Xcompiler=-fPIC", "-arch=sm_80", "--default-stream=per-thread")
        add_culdflags("-Xcompiler=-fPIC")
    end

    add_files("../src/ops/*/nvidia/*.cu")
    add_files("../src/ops/self_attention/nvidia/flashinfer_adapter.cu", {cuflags = "-w"})
    add_links("cublas", "cudart")

    on_install(function (target) end)
target_end()
