#pragma once

#include <torch/torch.h>
#include <string>
#include <vector>
#include <memory>
#include <cstdint>

/**
 * @brief 基于嵌入式 Python C-API 调用 tiktoken 实现的分词与解码器
 */
class TiktokenDecoder {
public:
    /**
     * @brief 使用 CMake 选中的 Python 环境初始化解释器并加载指定 encoding
     * @param encoding_name 编码器名称，默认 "gpt2"
     */
    explicit TiktokenDecoder(const std::string& encoding_name = "gpt2");

    /**
     * @brief 析构函数，释放本实例持有的 Python 对象引用
     */
    ~TiktokenDecoder();

    // 禁用拷贝语义，允许移动语义
    TiktokenDecoder(const TiktokenDecoder&) = delete;
    TiktokenDecoder& operator=(const TiktokenDecoder&) = delete;
    TiktokenDecoder(TiktokenDecoder&&) noexcept;
    TiktokenDecoder& operator=(TiktokenDecoder&&) noexcept;

    /**
     * @brief 检查 Python 及 tiktoken 编码器是否初始化成功
     */
    bool is_valid() const;

    /**
     * @brief 在所有 decoder 销毁且相关线程结束后，从初始化线程关闭自有 Python 运行时。
     * 已由其他宿主初始化的 Python 不受支持，也不会由本类关闭。
     */
    static bool shutdown_python();

    /**
     * @brief 将文本编码为 token ID 向量
     */
    std::vector<int64_t> encode(const std::string& text);

    /**
     * @brief 将 token ID 向量解码还原为原始文本
     */
    std::string decode(const std::vector<int64_t>& token_ids);

    /**
     * @brief 重载便捷方法：直接从 LibTorch Tensor (1D 或 2D) 解码为文本
     */
    std::string decode(const torch::Tensor& tensor);

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};
