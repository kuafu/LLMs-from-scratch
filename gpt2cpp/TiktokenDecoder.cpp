#include "TiktokenDecoder.h"
#include "PythonEnvironment.h"
#include <torch/torch.h>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>

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

namespace
{
struct PyDecref
{
    void operator()(PyObject* object) const noexcept { Py_XDECREF(object); }
};
using PyRef = std::unique_ptr<PyObject, PyDecref>;

class GilGuard
{
public:
    GilGuard() : state_(PyGILState_Ensure()) {}
    ~GilGuard() { PyGILState_Release(state_); }
    GilGuard(const GilGuard&) = delete;
    GilGuard& operator=(const GilGuard&) = delete;
private:
    PyGILState_STATE state_;
};

bool matches_selected_path(const char* name, const wchar_t* expected)
{
    PyObject* actual = PySys_GetObject(name); // Borrowed reference.
    PyRef expected_path(PyUnicode_FromWideChar(expected, -1));
    PyRef path_module(PyImport_ImportModule("os.path"));
    if (!actual || !PyUnicode_Check(actual) || !expected_path || !path_module)
    {
        if (PyErr_Occurred()) PyErr_Print();
        std::cerr << "[PythonRuntime] Cannot inspect sys." << name << ".\n";
        return false;
    }

    PyRef same(PyObject_CallMethod(path_module.get(), "samefile", "OO", actual, expected_path.get()));
    int matches = same ? PyObject_IsTrue(same.get()) : -1;
    if (matches != 1)
    {
        if (PyErr_Occurred()) PyErr_Print();
        const char* actual_text = PyUnicode_AsUTF8(actual);
        const char* expected_text = PyUnicode_AsUTF8(expected_path.get());
        std::cerr << "[PythonRuntime] sys." << name << " does not match the CMake-selected environment.\n"
                  << "  Actual: " << (actual_text ? actual_text : "<unavailable>") << "\n"
                  << "  Expected: " << (expected_text ? expected_text : "<unavailable>") << "\n";
        if (PyErr_Occurred()) PyErr_Clear();
        return false;
    }
    return true;
}

class PythonRuntime
{
public:
    PythonRuntime() : owner_thread_(std::this_thread::get_id())
    {
        // A host-owned interpreter cannot be reconfigured or safely finalized here.
        if (Py_IsInitialized())
        {
            std::cerr << "[PythonRuntime] Python is already initialized by another owner; "
                         "the CMake-selected environment cannot be applied.\n";
            return;
        }

        PyConfig config;
        PyConfig_InitPythonConfig(&config);
        PyStatus status = PyConfig_SetString(&config, &config.program_name, GPT_PYTHON_EXECUTABLE);
        if (!PyStatus_Exception(status))
            status = PyConfig_SetString(&config, &config.executable, GPT_PYTHON_EXECUTABLE);
        if (!PyStatus_Exception(status))
            status = Py_InitializeFromConfig(&config);

        if (PyStatus_Exception(status))
        {
            std::cerr << "[PythonRuntime] Cannot initialize the CMake-selected Python: "
                      << (status.err_msg ? status.err_msg : "unknown initialization error") << "\n";
            PyConfig_Clear(&config);
            // A partially initialized interpreter is not used or finalized here.
            // The standalone application exits with an error in this case.
            return;
        }
        PyConfig_Clear(&config);

        if (!matches_selected_path("executable", GPT_PYTHON_EXECUTABLE) ||
            !matches_selected_path("prefix", GPT_PYTHON_PREFIX))
        {
            Py_FinalizeEx();
            return;
        }

        // Release the initialization thread's GIL. Each decoder operation
        // acquires it independently, including reference cleanup.
        owner_state_ = PyEval_SaveThread();
        phase_ = Phase::ready;
    }

    bool acquire()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (phase_ != Phase::ready)
        {
            std::cerr << "[PythonRuntime] The configured Python runtime is not available.\n";
            return false;
        }
        ++clients_;
        return true;
    }

    void release()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        --clients_;
    }

    bool shutdown(bool& stopped)
    {
        stopped = false;
        PyThreadState* owner_state = nullptr;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (phase_ == Phase::failed || phase_ == Phase::stopped)
            {
                stopped = true;
                return true;
            }
            if (phase_ != Phase::ready || clients_ != 0 ||
                std::this_thread::get_id() != owner_thread_)
            {
                std::cerr << "[PythonRuntime] Shutdown requires the initialization thread, "
                             "with all decoders destroyed and worker threads stopped.\n";
                return false;
            }
            phase_ = Phase::stopping;
            owner_state = owner_state_;
            owner_state_ = nullptr;
        }

        // No runtime lock is held while entering or finalizing Python.
        PyEval_RestoreThread(owner_state);
        const int result = Py_FinalizeEx();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            phase_ = Phase::stopped;
        }
        stopped = true;
        if (result != 0) std::cerr << "[PythonRuntime] Python finalization failed.\n";
        return result == 0;
    }

private:
    enum class Phase { failed, ready, stopping, stopped };
    std::mutex mutex_;
    Phase phase_ = Phase::failed;
    std::size_t clients_ = 0;
    std::thread::id owner_thread_;
    PyThreadState* owner_state_ = nullptr;
};

struct RuntimeRegistry
{
    enum class Phase { idle, starting, ready, stopping, stopped };
    std::mutex mutex;
    std::condition_variable changed;
    Phase phase = Phase::idle;
    std::thread::id starting_thread;
    PythonRuntime* instance = nullptr;
};

RuntimeRegistry& runtime_registry()
{
    // The registry and runtime intentionally live until process exit. Python
    // itself is finalized explicitly, never during static destruction.
    static RuntimeRegistry* registry = new RuntimeRegistry();
    return *registry;
}

PythonRuntime* python_runtime()
{
    RuntimeRegistry& registry = runtime_registry();
    std::unique_lock<std::mutex> lock(registry.mutex);
    while (registry.phase == RuntimeRegistry::Phase::starting)
    {
        if (registry.starting_thread == std::this_thread::get_id())
        {
            std::cerr << "[PythonRuntime] Reentrant initialization is not supported.\n";
            return nullptr;
        }
        registry.changed.wait(lock);
    }
    if (registry.phase == RuntimeRegistry::Phase::ready) return registry.instance;
    if (registry.phase != RuntimeRegistry::Phase::idle)
    {
        std::cerr << "[PythonRuntime] The Python runtime has been shut down.\n";
        return nullptr;
    }

    registry.phase = RuntimeRegistry::Phase::starting;
    registry.starting_thread = std::this_thread::get_id();
    lock.unlock();
    PythonRuntime* instance = nullptr;
    try
    {
        instance = new PythonRuntime();
    }
    catch (...)
    {
        lock.lock();
        // A constructor exception may follow partial Python initialization.
        // Do not attempt a second initialization in this process.
        registry.phase = RuntimeRegistry::Phase::stopped;
        registry.changed.notify_all();
        throw;
    }
    lock.lock();
    registry.instance = instance;
    registry.phase = RuntimeRegistry::Phase::ready;
    registry.changed.notify_all();
    return instance;
}

bool shutdown_python_runtime()
{
    RuntimeRegistry& registry = runtime_registry();
    std::unique_lock<std::mutex> lock(registry.mutex);
    while (registry.phase == RuntimeRegistry::Phase::starting)
    {
        if (registry.starting_thread == std::this_thread::get_id())
        {
            std::cerr << "[PythonRuntime] Cannot shut down during initialization.\n";
            return false;
        }
        registry.changed.wait(lock);
    }
    if (registry.phase == RuntimeRegistry::Phase::idle)
    {
        registry.phase = RuntimeRegistry::Phase::stopped;
        return true;
    }
    if (registry.phase == RuntimeRegistry::Phase::stopped) return true;
    if (registry.phase != RuntimeRegistry::Phase::ready) return false;

    registry.phase = RuntimeRegistry::Phase::stopping;
    PythonRuntime* instance = registry.instance;
    lock.unlock();
    bool stopped = false;
    const bool result = instance->shutdown(stopped);
    lock.lock();
    registry.phase = stopped ? RuntimeRegistry::Phase::stopped : RuntimeRegistry::Phase::ready;
    registry.changed.notify_all();
    return result;
}
}

struct TiktokenDecoder::Impl
{
	PythonRuntime* runtime = nullptr;
	PyObject* pEncoding = nullptr;
	PyObject* pEncodeMethod = nullptr;
	PyObject* pDecodeMethod = nullptr;

	Impl(const std::string& encoding_name)
	{
		PythonRuntime* candidate = python_runtime();
		if (!candidate || !candidate->acquire()) return;
		runtime = candidate;

		try
		{
			GilGuard gil;
			PyRef module(PyImport_ImportModule("tiktoken"));
			if (!module)
			{
				PyErr_Print();
				std::cerr << "[TiktokenDecoder] Cannot import tiktoken in the CMake-selected Python environment.\n";
				return;
			}
			PyRef get_encoding(PyObject_GetAttrString(module.get(), "get_encoding"));
			if (!get_encoding || !PyCallable_Check(get_encoding.get()))
			{
				if (PyErr_Occurred()) PyErr_Print();
				std::cerr << "[TiktokenDecoder] tiktoken.get_encoding is not callable.\n";
				return;
			}
			PyRef name(PyUnicode_FromString(encoding_name.c_str()));
			PyRef args(name ? PyTuple_Pack(1, name.get()) : nullptr);
			PyRef encoding(args ? PyObject_CallObject(get_encoding.get(), args.get()) : nullptr);
			if (!encoding)
			{
				if (PyErr_Occurred()) PyErr_Print();
				std::cerr << "[TiktokenDecoder] Cannot load encoding: " << encoding_name << "\n";
				return;
			}
			PyRef encode(PyObject_GetAttrString(encoding.get(), "encode"));
			if (!encode || !PyCallable_Check(encode.get()))
			{
				if (PyErr_Occurred()) PyErr_Print();
				std::cerr << "[TiktokenDecoder] encoding.encode is not callable.\n";
				return;
			}
			PyRef decode(PyObject_GetAttrString(encoding.get(), "decode"));
			if (!decode || !PyCallable_Check(decode.get()))
			{
				if (PyErr_Occurred()) PyErr_Print();
				std::cerr << "[TiktokenDecoder] encoding.decode is not callable.\n";
				return;
			}

			PyObject* executable = PySys_GetObject("executable"); // Borrowed.
			PyObject* prefix = PySys_GetObject("prefix"); // Borrowed.
			PyRef module_file(PyObject_GetAttrString(module.get(), "__file__"));
			if (!module_file) PyErr_Clear();
			const char* executable_text = executable ? PyUnicode_AsUTF8(executable) : nullptr;
			const char* prefix_text = prefix ? PyUnicode_AsUTF8(prefix) : nullptr;
			const char* module_text = module_file ? PyUnicode_AsUTF8(module_file.get()) : nullptr;
			std::cout << "Tokenizer Python: " << (executable_text ? executable_text : "<unavailable>") << "\n"
			          << "Tokenizer environment: " << (prefix_text ? prefix_text : "<unavailable>") << "\n"
			          << "Tokenizer module: " << (module_text ? module_text : "<unavailable>") << "\n";
			if (PyErr_Occurred()) PyErr_Clear();

			pEncoding = encoding.release();
			pEncodeMethod = encode.release();
			pDecodeMethod = decode.release();
		}
		catch (...)
		{
			runtime->release();
			runtime = nullptr;
			throw;
		}
	}

	~Impl()
	{
		if (runtime)
		{
			{
				GilGuard gil;
				Py_XDECREF(pEncodeMethod);
				Py_XDECREF(pDecodeMethod);
				Py_XDECREF(pEncoding);
			}
			runtime->release();
		}
	}

	std::vector<int64_t> encode(const std::string& text)
	{
		std::vector<int64_t> tokens;
		if (!pEncodeMethod)
		{
			std::cerr << "[TiktokenDecoder] The encoder is not initialized.\n";
			return tokens;
		}

		GilGuard gil;
		PyRef pText(PyUnicode_FromString(text.c_str()));
		PyRef pArgs(pText ? PyTuple_Pack(1, pText.get()) : nullptr);
		PyRef pResult(pArgs ? PyObject_CallObject(pEncodeMethod, pArgs.get()) : nullptr);
		if (!pResult)
		{
			if (PyErr_Occurred()) PyErr_Print();
			std::cerr << "[TiktokenDecoder] Encoding failed.\n";
			return tokens;
		}

		if (!PyList_Check(pResult.get()))
		{
			std::cerr << "[TiktokenDecoder] Encoding did not return a list.\n";
			return tokens;
		}
		Py_ssize_t size = PyList_Size(pResult.get());
		tokens.reserve(static_cast<size_t>(size));
		for (Py_ssize_t i = 0; i < size; ++i)
		{
			PyObject* item = PyList_GetItem(pResult.get(), i); // Borrowed.
			const int64_t token = PyLong_AsLongLong(item);
			if (PyErr_Occurred())
			{
				PyErr_Print();
				return {};
			}
			tokens.push_back(token);
		}
		return tokens;
	}

	std::string decode(const std::vector<int64_t>& token_ids)
	{
		if (!pDecodeMethod)
		{
			std::cerr << "[TiktokenDecoder] The decoder is not initialized.\n";
			return {};
		}

		GilGuard gil;
		PyRef pList(PyList_New(static_cast<Py_ssize_t>(token_ids.size())));
		if (!pList)
		{
			PyErr_Print();
			return {};
		}
		for (size_t i = 0; i < token_ids.size(); ++i)
		{
			PyRef item(PyLong_FromLongLong(token_ids[i]));
			if (!item || PyList_SetItem(pList.get(), static_cast<Py_ssize_t>(i), item.release()) != 0)
			{
				if (PyErr_Occurred()) PyErr_Print();
				return {};
			}
		}

		PyRef pArgs(PyTuple_Pack(1, pList.get()));
		PyRef pResult(pArgs ? PyObject_CallObject(pDecodeMethod, pArgs.get()) : nullptr);
		if (!pResult)
		{
			if (PyErr_Occurred()) PyErr_Print();
			std::cerr << "[TiktokenDecoder] Decoding failed.\n";
			return {};
		}

		if (!PyUnicode_Check(pResult.get()))
		{
			std::cerr << "[TiktokenDecoder] Decoding did not return text.\n";
			return {};
		}
		const char* text = PyUnicode_AsUTF8(pResult.get());
		if (!text)
		{
			PyErr_Print();
			return {};
		}
		return text;
	}
};

TiktokenDecoder::TiktokenDecoder(const std::string& encoding_name)
	: pimpl(std::make_unique<Impl>(encoding_name))
{
}

TiktokenDecoder::~TiktokenDecoder() = default;
TiktokenDecoder::TiktokenDecoder(TiktokenDecoder&&) noexcept = default;
TiktokenDecoder& TiktokenDecoder::operator=(TiktokenDecoder&&) noexcept = default;

bool TiktokenDecoder::is_valid() const
{
	return pimpl && pimpl->pEncoding && pimpl->pEncodeMethod && pimpl->pDecodeMethod;
}

bool TiktokenDecoder::shutdown_python()
{
	return shutdown_python_runtime();
}

std::vector<int64_t> TiktokenDecoder::encode(const std::string& text)
{
	if (!pimpl) return {};
	return pimpl->encode(text);
}

std::string TiktokenDecoder::decode(const std::vector<int64_t>& token_ids)
{
	if (!pimpl) return "";
	return pimpl->decode(token_ids);
}

std::string TiktokenDecoder::decode(const torch::Tensor& tensor)
{
	auto cpu_tensor = tensor.detach().to(torch::kCPU).to(torch::kInt64).contiguous();
	if (cpu_tensor.dim() == 2)
	{
		cpu_tensor = cpu_tensor.squeeze(0);
	}
	std::vector<int64_t> token_ids(
		cpu_tensor.data_ptr<int64_t>(),
		cpu_tensor.data_ptr<int64_t>() + cpu_tensor.numel()
	);
	return decode(token_ids);
}
