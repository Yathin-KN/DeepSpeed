#pragma once

#include <dlfcn.h>
#include <iostream>
#include <memory>
#include <string>
#include "../include/deepspeed_aio_base.h"

class Trampoline {
public:
  explicit Trampoline(const std::string& device_type) : device(nullptr), handle(nullptr) {
    // Constructor that initializes the device and handle pointers to nullptr
    // and loads the device based on the provided device_type
    load_device(device_type);
}

void load_device(const std::string& device_type) {
    if (device) {
        // If a device is already loaded, delete it to avoid memory leaks
        delete device;
    }

    if (handle) {
        // If a handle to a shared library is already open, close it
        dlclose(handle);
    }

    // Construct the path to the shared library (.so file) for the device
    std::filesystem::path so_directory =
        std::filesystem::current_path() / "deepspeed" / "ops" / "plugins";

    // Create the full path to the shared library by appending the device type and file extension
    std::filesystem::path lib_path = so_directory / (device_type + "_op.so");

    // Open the shared library with dlopen
    handle = dlopen(lib_path.c_str(), RTLD_LAZY);
    if (!handle) {
        // If the library cannot be opened, print an error message and return
        std::cerr << "Cannot open library: " << dlerror() << '\n';
        return;
    }

    // Clear any existing errors
    dlerror();

    // Define a function pointer type for the create_device function
    typedef DeepSpeedAIOBase* (*create_t)();
    
    // Get the address of the create_device function from the shared library
    create_t create_device = (create_t)dlsym(handle, "create_device");
    const char* dlsym_error = dlerror();
    if (dlsym_error) {
        // If there is an error retrieving the symbol, print an error message,
        // close the handle, and set it to nullptr
        std::cerr << "Cannot load symbol create_device: " << dlsym_error << '\n';
        dlclose(handle);
        handle = nullptr;
        return;
    }

    // Call the create_device function to create an instance of the device
    device = create_device();
}

void aio_read(torch::Tensor& buffer, const char* filename, const bool validate) {
    if (device) {
        // If a device is loaded, perform the asynchronous read operation
        device->aio_read(buffer, filename, validate);
    } else {
        // If no device is loaded, print an error message
        std::cerr << "No device loaded for aio_read\n";
    }
}

    void aio_write(const torch::Tensor& buffer, const char* filename, const bool validate)
    {
        if (device)
            device->aio_write(buffer, filename, validate);
        else
            std::cerr << "No device loaded for aio_write\n";
    }
    void deepspeed_memcpy(torch::Tensor& dest, const torch::Tensor& src)
    {
        if (device)
            device->deepspeed_memcpy(dest, src);
        else
            std::cerr << "No device loaded for deepspeed_memcpy\n";
    }

    int get_block_size() const
    {
        if (device)
            return device->get_block_size();
        else {
            std::cerr << "No device loaded for get_block_size\n";
            return -1;
        }
    }
    int get_queue_depth() const
    {
        if (device)
            return device->get_queue_depth();
        else {
            std::cerr << "No device loaded for get_queue_depth\n";
            return -1;
        }
    }
    bool get_single_submit() const
    {
        if (device)
            return device->get_single_submit();
        else {
            std::cerr << "No device loaded for get_single_submit\n";
            return false;
        }
    }
    bool get_overlap_events() const
    {
        if (device)
            return device->get_overlap_events();
        else {
            std::cerr << "No device loaded for get_overlap_events\n";
            return false;
        }
    }
    int get_thread_count() const
    {
        if (device)
            return device->get_thread_count();
        else {
            std::cerr << "No device loaded for get_thread_count\n";
            return -1;
        }
    }

    void read(torch::Tensor& buffer, const char* filename, const bool validate)
    {
        if (device)
            device->read(buffer, filename, validate);
        else
            std::cerr << "No device loaded for read\n";
    }
    void write(const torch::Tensor& buffer, const char* filename, const bool validate)
    {
        if (device)
            device->write(buffer, filename, validate);
        else
            std::cerr << "No device loaded for write\n";
    }
    void pread(const torch::Tensor& buffer,
               const char* filename,
               const bool validate,
               const bool async)
    {
        if (device)
            device->pread(buffer, filename, validate, async);
        else
            std::cerr << "No device loaded for pread\n";
    }
    void pwrite(const torch::Tensor& buffer,
                const char* filename,
                const bool validate,
                const bool async)
    {
        if (device)
            device->pwrite(buffer, filename, validate, async);
        else
            std::cerr << "No device loaded for pwrite\n";
    }

    void sync_pread(torch::Tensor& buffer, const char* filename)
    {
        if (device)
            device->sync_pread(buffer, filename);
        else
            std::cerr << "No device loaded for sync_pread\n";
    }
    void sync_pwrite(const torch::Tensor& buffer, const char* filename)
    {
        if (device)
            device->sync_pwrite(buffer, filename);
        else
            std::cerr << "No device loaded for sync_pwrite\n";
    }
    void async_pread(torch::Tensor& buffer, const char* filename)
    {
        if (device)
            device->async_pread(buffer, filename);
        else
            std::cerr << "No device loaded for async_pread\n";
    }
    void async_pwrite(const torch::Tensor& buffer, const char* filename)
    {
        if (device)
            device->async_pwrite(buffer, filename);
        else
            std::cerr << "No device loaded for async_pwrite\n";
    }

    void new_cpu_locked_tensor(const size_t num_elem, const torch::Tensor& example_tensor)
    {
        if (device)
            device->new_cpu_locked_tensor(num_elem, example_tensor);
        else
            std::cerr << "No device loaded for new_cpu_locked_tensor\n";
    }
    void free_cpu_locked_tensor(torch::Tensor& tensor)
    {
        if (device)
            device->free_cpu_locked_tensor(tensor);
        else
            std::cerr << "No device loaded for free_cpu_locked_tensor\n";
    }

    void wait()
    {
        if (device)
            device->wait();
        else
            std::cerr << "No device loaded for wait\n";
    }

    ~Trampoline()
    {
        if (device) { delete device; }
        if (handle) { dlclose(handle); }
    }

private:
    DeepSpeedAIOBase* device;
    void* handle;
};
    }

private:
    DeepSpeedAIOBase* device;
    void* handle;
};



























