#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

//判断连续：通过从最后一维往前检查，步长是否满足维度末尾乘法规则----view和load只能在连续tensor上操作
bool Tensor::isContiguous() const {
    size_t z = 1;
    for (size_t i = _meta.shape.size(); i-- > 0;) {
        if (_meta.shape[i] != 1) {
            if (_meta.strides[i] != static_cast<ptrdiff_t>(z)) return false;
            z *= _meta.shape[i];
        }
    }
    return true;
}

//要求连续，只是重新解释形状
tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    if (order.size() != this->ndim()) {
        throw std::runtime_error("order不合法");
    }
    std::vector<size_t> new_shape;
    std::vector<ptrdiff_t> new_strides;
    for (size_t i : order) {
        new_shape.push_back(_meta.shape[i]);
        new_strides.push_back(_meta.strides[i]);
    }
    TensorMeta new_meta{_meta.dtype, new_shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}
//交换维度=交换 stride，数据不动
tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    if (!isContiguous()) {
        throw std::runtime_error("tensor不连续");
    }
    size_t new_numel = std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
    //标准库内accumulate：累积
    if (new_numel != this->numel()) {
        throw std::runtime_error("元素数量不同");
    }

    //数据没变，只是把shape改了后根据新shape创建新步长
    std::vector<ptrdiff_t> new_strides(shape.size());
    size_t stride = 1;
    for (size_t i = shape.size(); i-- > 0;) {
        new_strides[i] = static_cast<ptrdiff_t>(stride);
        stride *= shape[i];
    }

    TensorMeta new_meta = {_meta.dtype, shape, new_strides}; //保留了原数据_meta.dtype的情况下，使用新shape和new_strides
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}
//移动起始指针，缩小某一维的长度
tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    if (dim >= ndim()) {
        throw std::runtime_error("维度超过");
    }
    if (start >= end || end > _meta.shape[dim]) {
        throw std::runtime_error("范围超过");
    }
    TensorMeta new_meta = _meta;
    new_meta.shape[dim] = end - start;
    size_t added_offset = start * _meta.strides[dim] * elementSize(); //跳过 start 个位置 × 步长 × 字节数
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset + added_offset));
}
// 从 CPU 加载数据，这是权重加载的底层入口——Python 侧读 safetensors 文件到 CPU 内存，然后调 load() 传到 GPU
void Tensor::load(const void *src_) {
    if(!isContiguous()){
        throw std::runtime_error("tensor不连续");
    }
    size_t size_in_byte = numel() * elementSize(); //计算要多少字节 tensor中的元素数 * 字节数
    
    if(deviceType() == LLAISYS_DEVICE_CPU){
        std::memcpy(data(), src_, size_in_byte);
    }else{
        core::context().setDevice(deviceType(), deviceId());//显式设置当前 CPU 线程的上下文（Context），使其绑定到指定的硬件设备上。
        core::context().runtime().api()->memcpy_sync(
            data(),             // 目标：GPU 显存地址
            src_,                // 源：用户传入的 CPU 指针
            size_in_byte,      // 大小
            LLAISYS_MEMCPY_H2D  // 方向：主机到设备
        );
    }
}
namespace{
    void copy_strided_cpu(const std::byte* src, std::byte* dst, 
                          const std::vector<size_t>& shape, 
                          const std::vector<ptrdiff_t>& strides, 
                          size_t elem_size, size_t dim, size_t& dst_offset){ //dst_offset是写入字节的游标
        if(dim == shape.size() - 1){ //如果递归终止，进行拷贝数据
            for(size_t i = 0; i < shape[dim]; ++i){
                std::memcpy(dst + dst_offset, src + i * strides[dim] * elem_size, elem_size); //strides[dim] * elem_size当前维度的步长*字节数，src是起始位置，i是步数
                dst_offset += elem_size;//dst是起始位置，dst_offset是字节数，相当于src每一步跳strides[dim] * elem_size，但是dst只往下进行一次，顺序写入
            }
        }else{
            for(size_t i = 0; i < shape[dim]; ++i){
                copy_strided_cpu(src + i * strides[dim] * elem_size, dst, shape, strides, elem_size, dim + 1, dst_offset);
            }
        }
    }
}
//把不连续的变连续
tensor_t Tensor::contiguous() const {
    if(isContiguous()) return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));

    auto res = create(_meta.shape, _meta.dtype, deviceType(), deviceId());
    if(deviceType() == LLAISYS_DEVICE_CPU){ //cpu上用按 stride 逐元素拷贝到新的连续内存
        size_t dst_offset = 0; //游标从0开始
        copy_strided_cpu(this->data(), res->data(), _meta.shape, _meta.strides, elementSize(), 0, dst_offset); //这里dim用的0，是因为copy_strided_cpu是个递归函数，从 0 递归到 shape.size() - 1
    }else {
        //先 D2H 拷到 CPU → CPU 上做一次 contiguous → 再 H2D 传回 GPU
        size_t raw_size = _storage->size();
        auto cpu_storage = core::context().runtime().allocateHostStorage(raw_size);
        core::context().setDevice(deviceType(), deviceId());
        core::context().runtime().api()->memcpy_sync(
            cpu_storage->memory(), 
            _storage->memory(), 
            raw_size, 
            LLAISYS_MEMCPY_D2H
        );
        auto cpu_mirror = std::shared_ptr<Tensor>(new Tensor(_meta, cpu_storage, _offset));
        auto cpu_contig = cpu_mirror->contiguous();
        res = cpu_contig->to(deviceType(), deviceId());

    }
    return res;
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    return this -> contiguous() -> view(shape);
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    if(!isContiguous()){
        return this -> contiguous() -> to(device_type, device);
    }
    auto res = Tensor::create(_meta.shape, _meta.dtype, device_type, device);
    llaisysMemcpyKind_t kind;
    if (deviceType() == LLAISYS_DEVICE_CPU && device_type != LLAISYS_DEVICE_CPU) {
        kind = LLAISYS_MEMCPY_H2D; // CPU -> GPU
    } else if (deviceType() != LLAISYS_DEVICE_CPU && device_type == LLAISYS_DEVICE_CPU) {
        kind = LLAISYS_MEMCPY_D2H; // GPU -> CPU
    } else {
        kind = LLAISYS_MEMCPY_D2D; // GPU -> GPU (或者 CPU->CPU)
    }
    // 切换到目标设备的上下文（最佳实践：通常在目标设备上发起接收指令比较稳妥）
    core::context().setDevice(device_type, device);
    
    // 计算大小
    size_t size_in_bytes = numel() * elementSize();

    core::context().runtime().api()->memcpy_sync(
        res->data(),    // 目标地址
        this->data(),   // 源地址
        size_in_bytes,  // 字节数
        kind            // 方向
    );
    return res;
}

} // namespace llaisys
