#include <torch/torch.h>
#include "TiktokenDecoder.h"
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <limits>
#include <iomanip>

////////////////////////////////////////////////////////////////////////////////
// Configuration
////////////////////////////////////////////////////////////////////////////////
struct GPTConfig {
    int64_t vocab_size = 50257;     // 词表大小
    int64_t context_length = 1024;  // 上下文长度
    int64_t emb_dim = 768;          // 嵌入维度
    int64_t n_heads = 12;           // 注意力头数
    int64_t n_layers = 12;          // Transformer层数
    double drop_rate = 0.1;         // Dropout比率
    bool qkv_bias = false;          // QKV线性层偏置
};

////////////////////////////////////////////////////////////////////////////////
// Chapter 2: Dataset
////////////////////////////////////////////////////////////////////////////////
class GPTDatasetV1 : public torch::data::Dataset<GPTDatasetV1> {
private:
    std::vector<torch::Tensor> input_ids;
    std::vector<torch::Tensor> target_ids;

public:
    GPTDatasetV1(const std::vector<int64_t>& token_ids, int64_t max_length, int64_t stride) {
        if (token_ids.size() > static_cast<size_t>(max_length)) {
            for (size_t i = 0; i <= token_ids.size() - max_length - 1; i += stride) {
                auto input_chunk = std::vector<int64_t>(
                    token_ids.begin() + i,
                    token_ids.begin() + i + max_length
                );
                auto target_chunk = std::vector<int64_t>(
                    token_ids.begin() + i + 1,
                    token_ids.begin() + i + max_length + 1
                );
                input_ids.push_back(torch::tensor(input_chunk, torch::kInt64));
                target_ids.push_back(torch::tensor(target_chunk, torch::kInt64));
            }
        }
    }

    torch::data::Example<> get(size_t index) override {
        return {input_ids[index], target_ids[index]};
    }

    torch::optional<size_t> size() const override {
        return input_ids.size();
    }
};

////////////////////////////////////////////////////////////////////////////////
// Chapter 3: Multi-Head Attention
////////////////////////////////////////////////////////////////////////////////
class MultiHeadAttentionImpl : public torch::nn::Module {
private:
    int64_t d_out;
    int64_t num_heads;
    int64_t head_dim;

    torch::nn::Linear W_query{nullptr};
    torch::nn::Linear W_key{nullptr};
    torch::nn::Linear W_value{nullptr};
    torch::nn::Linear out_proj{nullptr};
    torch::nn::Dropout dropout{nullptr};
    torch::Tensor mask;

public:
    MultiHeadAttentionImpl(int64_t d_in, int64_t d_out, int64_t context_length,
                           double dropout_rate, int64_t num_heads, bool qkv_bias = false)
        : d_out(d_out), num_heads(num_heads), head_dim(d_out / num_heads) {
        TORCH_CHECK(d_out % num_heads == 0, "d_out must be divisible by num_heads");

        W_query = register_module("W_query", torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias)));
        W_key = register_module("W_key", torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias)));
        W_value = register_module("W_value", torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias)));
        out_proj = register_module("out_proj", torch::nn::Linear(d_out, d_out));
        dropout = register_module("dropout", torch::nn::Dropout(dropout_rate));

        // 注册上三角掩码缓存 (mask)
        mask = register_buffer("mask", torch::triu(torch::ones({context_length, context_length}), 1));
    }

    torch::Tensor forward(torch::Tensor x) {
        auto b = x.size(0);
        auto num_tokens = x.size(1);
        auto d_in = x.size(2);

        auto keys = W_key(x);
        auto queries = W_query(x);
        auto values = W_value(x);

        // 拆分多头并转置: (b, num_tokens, d_out) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.view({b, num_tokens, num_heads, head_dim}).transpose(1, 2);
        queries = queries.view({b, num_tokens, num_heads, head_dim}).transpose(1, 2);
        values = values.view({b, num_tokens, num_heads, head_dim}).transpose(1, 2);

        // 计算注意力分数: Q @ K^T
        auto attn_scores = torch::matmul(queries, keys.transpose(2, 3));

        // 截取当前 token 长度的掩码并转换为布尔型
        auto mask_bool = mask.to(torch::kBool).slice(0, 0, num_tokens).slice(1, 0, num_tokens);

        // 填充负无穷
        attn_scores.masked_fill_(mask_bool, -std::numeric_limits<float>::infinity());

        // Softmax 归一化缩放
        double scale = std::sqrt(static_cast<double>(keys.size(-1)));
        auto attn_weights = torch::softmax(attn_scores / scale, -1);
        attn_weights = dropout(attn_weights);

        // 加权求和上下文向量
        auto context_vec = torch::matmul(attn_weights, values).transpose(1, 2);
        context_vec = context_vec.contiguous().view({b, num_tokens, d_out});
        context_vec = out_proj(context_vec);

        return context_vec;
    }
};
TORCH_MODULE(MultiHeadAttention);

////////////////////////////////////////////////////////////////////////////////
// Chapter 4: Components & GPT Model
////////////////////////////////////////////////////////////////////////////////
class LayerNormImpl : public torch::nn::Module {
private:
    double eps;
    torch::Tensor scale;
    torch::Tensor shift;

public:
    LayerNormImpl(int64_t emb_dim, double eps = 1e-5) : eps(eps) {
        scale = register_parameter("scale", torch::ones({emb_dim}));
        shift = register_parameter("shift", torch::zeros({emb_dim}));
    }

    torch::Tensor forward(torch::Tensor x) {
        auto mean = x.mean(-1, /*keepdim=*/true);
        // unbiased=false 对应 correction=0
        auto var = x.var(-1, /*unbiased=*/false, /*keepdim=*/true);
        auto norm_x = (x - mean) / torch::sqrt(var + eps);
        return scale * norm_x + shift;
    }
};
TORCH_MODULE(LayerNorm);

class GELUImpl : public torch::nn::Module {
public:
    GELUImpl() = default;

    torch::Tensor forward(torch::Tensor x) {
        const double sqrt_2_over_pi = std::sqrt(2.0 / 3.14159265358979323846);
        return 0.5 * x * (1.0 + torch::tanh(
            sqrt_2_over_pi * (x + 0.044715 * torch::pow(x, 3))
        ));
    }
};
TORCH_MODULE(GELU);

class FeedForwardImpl : public torch::nn::Module {
private:
    torch::nn::Sequential layers{nullptr};

public:
    FeedForwardImpl(const GPTConfig& cfg) {
        layers = register_module("layers", torch::nn::Sequential(
            torch::nn::Linear(cfg.emb_dim, 4 * cfg.emb_dim),
            GELU(),
            torch::nn::Linear(4 * cfg.emb_dim, cfg.emb_dim)
        ));
    }

    torch::Tensor forward(torch::Tensor x) {
        return layers->forward(x);
    }
};
TORCH_MODULE(FeedForward);

class TransformerBlockImpl : public torch::nn::Module {
private:
    MultiHeadAttention att{nullptr};
    FeedForward ff{nullptr};
    LayerNorm norm1{nullptr};
    LayerNorm norm2{nullptr};
    torch::nn::Dropout drop_shortcut{nullptr};

public:
    TransformerBlockImpl(const GPTConfig& cfg) {
        att = register_module("att", MultiHeadAttention(
            cfg.emb_dim,
            cfg.emb_dim,
            cfg.context_length,
            cfg.drop_rate,
            cfg.n_heads,
            cfg.qkv_bias
        ));
        ff = register_module("ff", FeedForward(cfg));
        norm1 = register_module("norm1", LayerNorm(cfg.emb_dim));
        norm2 = register_module("norm2", LayerNorm(cfg.emb_dim));
        drop_shortcut = register_module("drop_shortcut", torch::nn::Dropout(cfg.drop_rate));
    }

    torch::Tensor forward(torch::Tensor x) {
        // 自注意力残差连接
        auto shortcut = x;
        x = norm1(x);
        x = att(x);
        x = drop_shortcut(x);
        x = x + shortcut;

        // 前馈网络残差连接
        shortcut = x;
        x = norm2(x);
        x = ff(x);
        x = drop_shortcut(x);
        x = x + shortcut;

        return x;
    }
};
TORCH_MODULE(TransformerBlock);

class GPTModelImpl : public torch::nn::Module {
private:
    torch::nn::Embedding tok_emb{nullptr};
    torch::nn::Embedding pos_emb{nullptr};
    torch::nn::Dropout drop_emb{nullptr};
    torch::nn::Sequential trf_blocks{nullptr};
    LayerNorm final_norm{nullptr};
    torch::nn::Linear out_head{nullptr};

public:
    GPTModelImpl(const GPTConfig& cfg) {
        tok_emb = register_module("tok_emb", torch::nn::Embedding(cfg.vocab_size, cfg.emb_dim));
        pos_emb = register_module("pos_emb", torch::nn::Embedding(cfg.context_length, cfg.emb_dim));
        drop_emb = register_module("drop_emb", torch::nn::Dropout(cfg.drop_rate));

        trf_blocks = register_module("trf_blocks", torch::nn::Sequential());
        for (int64_t i = 0; i < cfg.n_layers; ++i) {
            trf_blocks->push_back(TransformerBlock(cfg));
        }

        final_norm = register_module("final_norm", LayerNorm(cfg.emb_dim));
        out_head = register_module("out_head", torch::nn::Linear(
            torch::nn::LinearOptions(cfg.emb_dim, cfg.vocab_size).bias(false)
        ));
    }

    torch::Tensor forward(torch::Tensor in_idx) {
        auto batch_size = in_idx.size(0);
        auto seq_len = in_idx.size(1);

        auto tok_embeds = tok_emb(in_idx);
        auto pos_embeds = pos_emb(torch::arange(seq_len, in_idx.options()));
        auto x = tok_embeds + pos_embeds;
        x = drop_emb(x);
        x = trf_blocks->forward(x);
        x = final_norm(x);
        auto logits = out_head(x);
        return logits;
    }
};
TORCH_MODULE(GPTModel);

////////////////////////////////////////////////////////////////////////////////
// Text Generation Function
////////////////////////////////////////////////////////////////////////////////
torch::Tensor generate_text_simple(GPTModel& model, torch::Tensor idx, int64_t max_new_tokens, int64_t context_size) {
    for (int64_t i = 0; i < max_new_tokens; ++i) {
        int64_t seq_len = idx.size(1);
        int64_t start_col = std::max<int64_t>(0, seq_len - context_size);
        auto idx_cond = idx.slice(/*dim=*/1, start_col, seq_len);

        torch::Tensor logits;
        {
            torch::NoGradGuard no_grad;
            logits = model->forward(idx_cond);
        }

        // 只保留最后一个时间步: (batch, n_token, vocab_size) -> (batch, vocab_size)
        logits = logits.select(/*dim=*/1, -1);

        // 选出最大 logit 的 token id: (batch, 1)
        auto idx_next = torch::argmax(logits, /*dim=*/-1, /*keepdim=*/true);

        // 拼接到当前序列末尾
        idx = torch::cat({idx, idx_next}, /*dim=*/1);
    }
    return idx;
}

////////////////////////////////////////////////////////////////////////////////
// Main Function
////////////////////////////////////////////////////////////////////////////////
int main() {
    GPTConfig cfg_124m;
    cfg_124m.vocab_size = 50257;
    cfg_124m.context_length = 1024;
    cfg_124m.emb_dim = 768;
    cfg_124m.n_heads = 12;
    cfg_124m.n_layers = 12;
    cfg_124m.drop_rate = 0.1;
    cfg_124m.qkv_bias = false;

    torch::manual_seed(123);
    GPTModel model(cfg_124m);
    model->eval(); // 禁用 dropout

    std::string start_context = "Hello, I am";

    TiktokenDecoder tokenizer("gpt2");
    std::vector<int64_t> encoded = tokenizer.encode(start_context);
    if (encoded.empty()) {
        encoded = {15496, 11, 314, 716};
    }
    auto encoded_tensor = torch::tensor(encoded, torch::kInt64).unsqueeze(0);

    std::cout << "\n" << std::string(50, '=') << "\n"
              << std::string(22, ' ') << "IN\n"
              << std::string(50, '=') << "\n\n";
    std::cout << "Input text: " << start_context << "\n";
    std::cout << "Encoded input text: [";
    for (size_t i = 0; i < encoded.size(); ++i) {
        std::cout << encoded[i] << (i + 1 < encoded.size() ? ", " : "");
    }
    std::cout << "]\n";
    std::cout << "encoded_tensor.shape: [" << encoded_tensor.size(0) << ", " << encoded_tensor.size(1) << "]\n";

    auto out = generate_text_simple(
        model,
        encoded_tensor,
        /*max_new_tokens=*/6,
        /*context_size=*/cfg_124m.context_length
    );

    std::string decoded_text = tokenizer.decode(out);

    std::cout << "\n\n" << std::string(50, '=') << "\n"
              << std::string(22, ' ') << "OUT\n"
              << std::string(50, '=') << "\n\n";
    std::cout << "Output: \n" << out << "\n";
    std::cout << "Output length: " << out.size(1) << "\n";
    std::cout << "Output text: " << decoded_text << "\n";

    return 0;
}
