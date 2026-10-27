#include "TiktokenDecoder.h"
#include <torch/torch.h>
#include <iostream>

// 规避 MSVC 在 Debug 模式下强制寻找不存在的 python3xx_d.lib
#ifdef _MSC_VER
#if defined(_DEBUG)
#undef _DEBUG
#include <Python.h>
#define _DEBUG
#else
#include <Python.h>
#endif
#else
#include <Python.h>
#endif

struct TiktokenDecoder::Impl {
    bool initialized_python_here = false;
    PyObject* pEncoding = nullptr;
    PyObject* pEncodeMethod = nullptr;
    PyObject* pDecodeMethod = nullptr;

    Impl(const std::string& encoding_name) {
        if (!Py_IsInitialized()) {
            Py_Initialize();
            initialized_python_here = true;
        }

        // 导入 tiktoken 模块
        PyObject* pModuleName = PyUnicode_FromString("tiktoken");
        PyObject* pModule = PyImport_Import(pModuleName);
        Py_DECREF(pModuleName);

        if (!pModule) {
            PyErr_Print();
            std::cerr << "[TiktokenDecoder] Error: 无法导入 Python 模块 'tiktoken'。请确保环境中已执行 pip install tiktoken。\n";
            return;
        }

        // 获取 tiktoken.get_encoding
        PyObject* pGetEncodingFunc = PyObject_GetAttrString(pModule, "get_encoding");
        Py_DECREF(pModule);

        if (!pGetEncodingFunc || !PyCallable_Check(pGetEncodingFunc)) {
            PyErr_Print();
            std::cerr << "[TiktokenDecoder] Error: tiktoken.get_encoding 不可调用。\n";
            Py_XDECREF(pGetEncodingFunc);
            return;
        }

        // 调用 get_encoding(encoding_name)
        PyObject* pArg = PyUnicode_FromString(encoding_name.c_str());
        PyObject* pArgs = PyTuple_Pack(1, pArg);
        Py_DECREF(pArg);

        pEncoding = PyObject_CallObject(pGetEncodingFunc, pArgs);
        Py_DECREF(pArgs);
        Py_DECREF(pGetEncodingFunc);

        if (!pEncoding) {
            PyErr_Print();
            std::cerr << "[TiktokenDecoder] Error: 调用 tiktoken.get_encoding(\"" << encoding_name << "\") 失败。\n";
            return;
        }

        // 获取 encode 和 decode 方法
        pEncodeMethod = PyObject_GetAttrString(pEncoding, "encode");
        pDecodeMethod = PyObject_GetAttrString(pEncoding, "decode");

        if (!pDecodeMethod || !PyCallable_Check(pDecodeMethod)) {
            PyErr_Print();
            std::cerr << "[TiktokenDecoder] Error: encoding.decode 方法不可调用。\n";
        }
    }

    ~Impl() {
        Py_XDECREF(pEncodeMethod);
        Py_XDECREF(pDecodeMethod);
        Py_XDECREF(pEncoding);

        if (initialized_python_here && Py_IsInitialized()) {
            Py_Finalize();
        }
    }

    std::vector<int64_t> encode(const std::string& text) {
        std::vector<int64_t> tokens;
        if (!pEncodeMethod) {
            std::cerr << "[TiktokenDecoder] encode 方法未就绪。\n";
            return tokens;
        }

        PyObject* pText = PyUnicode_FromString(text.c_str());
        PyObject* pArgs = PyTuple_Pack(1, pText);
        Py_DECREF(pText);

        PyObject* pResult = PyObject_CallObject(pEncodeMethod, pArgs);
        Py_DECREF(pArgs);

        if (!pResult) {
            PyErr_Print();
            return tokens;
        }

        if (PyList_Check(pResult)) {
            Py_ssize_t size = PyList_Size(pResult);
            tokens.reserve(size);
            for (Py_ssize_t i = 0; i < size; ++i) {
                PyObject* item = PyList_GetItem(pResult, i); // Borrowed reference
                tokens.push_back(PyLong_AsLongLong(item));
            }
        }
        Py_DECREF(pResult);
        return tokens;
    }

    std::string decode(const std::vector<int64_t>& token_ids) {
        if (!pDecodeMethod) {
            return "[TiktokenDecoder Error: 解码器未成功初始化]";
        }

        // 构造 Python List
        PyObject* pList = PyList_New(token_ids.size());
        for (size_t i = 0; i < token_ids.size(); ++i) {
            // PyList_SetItem 会窃取传入对象的引用 (steal reference)
            PyList_SetItem(pList, i, PyLong_FromLongLong(token_ids[i]));
        }

        PyObject* pArgs = PyTuple_Pack(1, pList);
        Py_DECREF(pList);

        PyObject* pResult = PyObject_CallObject(pDecodeMethod, pArgs);
        Py_DECREF(pArgs);

        if (!pResult) {
            PyErr_Print();
            return "[TiktokenDecoder Error: 解码调用失败]";
        }

        std::string decoded_str;
        if (PyUnicode_Check(pResult)) {
            const char* utf8_str = PyUnicode_AsUTF8(pResult);
            if (utf8_str) {
                decoded_str = utf8_str;
            }
        }
        Py_DECREF(pResult);
        return decoded_str;
    }
};

TiktokenDecoder::TiktokenDecoder(const std::string& encoding_name)
    : pimpl(std::make_unique<Impl>(encoding_name)) {}

TiktokenDecoder::~TiktokenDecoder() = default;
TiktokenDecoder::TiktokenDecoder(TiktokenDecoder&&) noexcept = default;
TiktokenDecoder& TiktokenDecoder::operator=(TiktokenDecoder&&) noexcept = default;

bool TiktokenDecoder::is_valid() const {
    return pimpl && pimpl->pDecodeMethod != nullptr;
}

std::vector<int64_t> TiktokenDecoder::encode(const std::string& text) {
    if (!pimpl) return {};
    return pimpl->encode(text);
}

std::string TiktokenDecoder::decode(const std::vector<int64_t>& token_ids) {
    if (!pimpl) return "";
    return pimpl->decode(token_ids);
}

std::string TiktokenDecoder::decode(const torch::Tensor& tensor) {
    auto cpu_tensor = tensor.detach().to(torch::kCPU).to(torch::kInt64).contiguous();
    if (cpu_tensor.dim() == 2) {
        cpu_tensor = cpu_tensor.squeeze(0);
    }
    std::vector<int64_t> token_ids(
        cpu_tensor.data_ptr<int64_t>(),
        cpu_tensor.data_ptr<int64_t>() + cpu_tensor.numel()
    );
    return decode(token_ids);
}
