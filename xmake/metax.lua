-- =============================================================
-- 沐曦 (MetaX) C500 GPU 构建规则
-- =============================================================
-- 用法: xmake build --metax-gpu=true
--
-- 当 MACA SDK 可用时，追加 ENABLE_METAX_RUNTIME 以启用真实实现。
-- 否则仅编译 stub 骨架（getDeviceCount 返回 0，其余抛异常）。
-- =============================================================

-- MACA SDK 路径检测
local maca_sdk = os.getenv("MACA_PATH") or "/opt/maca"
local maca_sdk_found = os.isdir(path.join(maca_sdk, "include"))
local mxcc = path.join(maca_sdk, "mxgpu_llvm/bin/mxcc")

target("llaisys-device-metax")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    add_defines("ENABLE_METAX_API")

    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("../src/device/metax/*.cpp")

    -- 如果 MACA SDK 存在，启用真实 runtime 实现
    if maca_sdk_found then
        add_defines("ENABLE_METAX_RUNTIME")
        add_includedirs(path.join(maca_sdk, "include"))
        add_includedirs(path.join(maca_sdk, "include/mcr"))
        add_includedirs(path.join(maca_sdk, "include/common"))
        add_linkdirs(path.join(maca_sdk, "lib"))
        add_links("mcruntime")
    end

    on_install(function (target) end)
target_end()

-- 沐曦算子库
target("llaisys-ops-metax")
    set_kind("static")
    add_deps("llaisys-tensor")
    add_defines("ENABLE_METAX_API")

    if maca_sdk_found then
        add_defines("ENABLE_METAX_RUNTIME")

        -- 使用 on_build 自定义编译，绕过 xmake 内置 CUDA 规则
        on_build(function (target)
            import("core.project.project")
            local mc_files = os.files(path.join(os.projectdir(), "src/ops/*/metax/*.mc"))
            local inc_dirs = {
                path.join(os.projectdir(), "include"),
                path.join(get_config("llmops-root") or "../llmops", "include"),
                path.join(maca_sdk, "include"),
                path.join(maca_sdk, "include/mcr"),
                path.join(maca_sdk, "include/common"),
                path.join(maca_sdk, "include/mcblas"),
                path.join(maca_sdk, "include/mcrand"),
            }
            local obj_dir = path.join(os.projectdir(), ".build/metax/objects")
            local obj_files = {}

            for _, src in ipairs(mc_files) do
                local obj_name = path.basename(src) .. ".o"
                local obj_path = path.join(obj_dir, obj_name)
                os.mkdir(obj_dir)

                -- 构建 mxcc 编译命令
                local args = {
                    "--cuda-gpu-arch=xcore1000",
                    "-x", "maca",
                    "-fPIC",
                    "-std=c++17",
                    "-O2",
                    "-DENABLE_METAX_API",
                    "-DENABLE_METAX_RUNTIME",
                    "-c", src,
                    "-o", obj_path,
                }
                for _, dir in ipairs(inc_dirs) do
                    table.insert(args, "-I" .. dir)
                end

                print("compiling.maca %s", path.relative(src))
                os.execv(mxcc, args)
                table.insert(obj_files, obj_path)
            end

            -- 打包为静态库
            local lib_path = path.join(os.projectdir(), ".build/metax/libllaisys-ops-metax.a")
            os.mkdir(path.directory(lib_path))
            os.execv("ar", table.join({"cr", lib_path}, obj_files))
            cprint("${green}archiving.release libllaisys-ops-metax.a")
        end)
    else
        -- 在非 MetaX 平台上编译桩实现（提供符号定义，运行时抛异常）
        set_languages("cxx17")
        set_warnings("all", "error")
        if not is_plat("windows") then
            add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        end
        add_files("../src/ops/metax_stubs.cpp")
    end

    on_install(function (target) end)
target_end()
