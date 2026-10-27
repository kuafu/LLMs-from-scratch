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
     * @brief 构造函数，初始化 Python 解释器并加载指定 encoding
     * @param encoding_name 编码器名称，默认 "gpt2"
     */
    explicit TiktokenDecoder(const std::string& encoding_name = "gpt2");

    /**
     * @brief 析构函数，释放 Python 对象引用并管理运行时生命周期
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
