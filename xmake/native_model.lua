option("rapidjson-include")
    set_default("")
    set_showmenu(true)
    set_description("RapidJSON include root for the optional native V4 config/checkpoint loader")
option_end()

-- Backend-independent model composition; linking TileLang is the caller's
-- explicit backend choice, not a dependency imposed by the model library.
if has_config("tilelang-native") or has_config("aten-native") then
    target("llaisys-deepseek-v4-native")
        set_kind("static")
        set_default(false)
        set_languages("cxx17")
        set_warnings("all", "error")
        add_cxflags("-fPIC")
        add_deps("llaisys-native-tensor")
        add_includedirs("..", {public = true})
        add_files("../src/models/deepseek_v4/linear.cpp", "../src/models/deepseek_v4/moe.cpp",
                  "../src/models/deepseek_v4/compressor.cpp", "../src/models/deepseek_v4/indexer.cpp",
                  "../src/models/deepseek_v4/attention.cpp", "../src/models/deepseek_v4/hyperconnection.cpp",
                  "../src/models/deepseek_v4/block.cpp")
        local json_root = get_config("rapidjson-include")
        if json_root and json_root ~= "" then
            add_includedirs(json_root, {system = true})
            add_files("../src/models/deepseek_v4/config.cpp", "../src/models/deepseek_v4/checkpoint.cpp", "../src/models/deepseek_v4/model.cpp",
                      "../src/models/deepseek_v4/session.cpp")
            add_syslinks("pthread", {public = true})
        end
    target_end()
end

if has_config("aten-native") and has_config("tilelang-native") and get_config("rapidjson-include") ~= "" then
    target("llaisys-v4-reference")
        set_kind("static")
        set_default(false)
        set_languages("cxx17")
        set_warnings("all", "error")
        add_cxflags("-fPIC")
        add_deps("llaisys-deepseek-v4-native", "llaisys-aten-native", "llaisys-tilelang-native")
        add_includedirs("..", {public = true})
        add_files("../src/backends/v4_reference.cpp")
    target_end()
    if has_config("python-bindings") then
        target("llaisys-v4-python")
            set_kind("shared")
            set_default(false)
            set_languages("cxx17")
            add_deps("llaisys-v4-reference")
            -- Reuse the audited installed runtime, without triggering its
            -- install hook or changing the Qwen baseline library.
            local runtime_lib = path.join(os.projectdir(), "python/llaisys/libllaisys")
            add_linkdirs(runtime_lib)
            add_links("llaisys")
            -- xmake places shflags after add_links; repeat these runtime
            -- registrations AFTER --no-as-needed so ELF keeps DT_NEEDED.
            add_shflags("-Wl,--no-as-needed,-lllaisys,-ltorch_cuda,-ltilelang", {force = true})
            add_rpathdirs(runtime_lib)
            add_includedirs("..", get_config("python-include"), get_config("pybind11-include"))
            add_files("../python/bindings/v4_native.cpp")
            -- Build only: do not overwrite the installed first-stage _C.so.
            set_filename("_v4_native.so")
        target_end()
    end
end
