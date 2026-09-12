option("tilelang-native")
    set_default(false)
    set_showmenu(true)
    set_description("Build the optional native TileLang backend and V4 quantized Linear")
option_end()

option("tilelang-root")
    set_default("")
    set_showmenu(true)
    set_description("Installed TileLang package root containing lib/libtilelang.so")
option_end()

option("tvm-ffi-root")
    set_default("")
    set_showmenu(true)
    set_description("Compatible TVM-FFI package root containing include/ and lib/")
option_end()

if has_config("tilelang-native") then
    local tl_root = get_config("tilelang-root")
    local ffi_root = get_config("tvm-ffi-root")

    target("llaisys-tilelang-native")
        set_kind("static")
        set_default(false)
        set_languages("cxx17")
        set_warnings("all", "error")
        add_cxflags("-fPIC")
        add_deps("llaisys-native-tensor")
        on_load(function (target)
            if not has_config("nv-gpu") then
                raise("tilelang-native currently requires --nv-gpu=y")
            end
            if not os.isfile(path.join(tl_root, "lib/libtilelang.so"))
                or not os.isfile(path.join(tl_root, "lib/libtvm.so")) then
                raise("tilelang-native requires --tilelang-root=<installed TileLang package>")
            end
            if not os.isfile(path.join(ffi_root, "include/tvm/ffi/function.h"))
                or not os.isfile(path.join(ffi_root, "include/dlpack/dlpack.h"))
                or not os.isfile(path.join(ffi_root, "lib/libtvm_ffi.so")) then
                raise("tilelang-native requires --tvm-ffi-root=<compatible TVM-FFI package>")
            end
        end)
        add_includedirs("..", {public = true})
        add_includedirs(path.join(ffi_root, "include"), {system = true, public = true})
        add_includedirs(path.join(tl_root, "3rdparty/tvm/include"), {system = true})
        add_files("../src/backends/tilelang/native_kernel.cpp")
        add_linkdirs(path.join(tl_root, "lib"), path.join(ffi_root, "lib"), {public = true})
        -- Registration/error helpers live in libtilelang; keep its initializer
        -- even when no symbol is directly referenced by the caller.
        add_ldflags("-Wl,--no-as-needed", {force = true, public = true})
        add_links("tilelang", "tvm", "tvm_ffi", "dl", {public = true})
        add_rpathdirs(path.join(ffi_root, "lib"), path.join(tl_root, "lib"), {public = true})
    target_end()
end
