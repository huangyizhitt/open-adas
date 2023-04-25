/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include "plugin_common.hpp"
#include "serialize.hpp"
#include "onnx2trt_common.hpp"

#include <NvInferPlugin.h>
#include <NvInferRuntimeCommon.h>
#include <memory>
#include <vector>

namespace onnx2trt {

// A convenient base class for plugins. Provides default implementations of
// some methods.
// Adapts a plugin so that its type is automatically serialized, enabling it
// to be identified when deserializing.
class Plugin : public nvinfer1::IPluginV2Ext, public IOwnable {
public:
  virtual const char* getPluginType() const noexcept = 0;

  nvinfer1::Dims const&  getInputDims(int index) const { return _input_dims.at(index); }
  size_t                 getMaxBatchSize()       const { return _max_batch_size; }
  nvinfer1::DataType     getDataType()           const { return _data_type; }
  nvinfer1::PluginFormat getDataFormat()         const { return _data_format; }

  size_t getWorkspaceSize(int) const noexcept override { return 0; }

  int         initialize() noexcept          override { return 0;}
  void        terminate() noexcept           override {}

  bool supportsFormat(nvinfer1::DataType type,
                      nvinfer1::PluginFormat format) const noexcept override;

  void configureWithFormat(const nvinfer1::Dims* inputDims, int nbInputs,
                           const nvinfer1::Dims* outputDims, int nbOutputs,
                           nvinfer1::DataType type,
                           nvinfer1::PluginFormat format,
                           int maxBatchSize) noexcept override;
  void destroy() noexcept override { delete this; }
protected:
  void   deserializeBase(void const*& serialData, size_t& serialLength);
  size_t getBaseSerializationSize() const;
  void   serializeBase(void*& buffer) const;

  std::vector<nvinfer1::Dims> _input_dims;
  size_t                      _max_batch_size;
  nvinfer1::DataType          _data_type;
  nvinfer1::PluginFormat      _data_format;
  virtual ~Plugin() {}
};

class PluginAdapter : public Plugin {
protected:
  nvinfer1::IPluginV2*     _plugin;
  nvinfer1::IPluginV2Ext*  _ext;
public:
  PluginAdapter(nvinfer1::IPluginV2* plugin) :
    _plugin(plugin), _ext(dynamic_cast<IPluginV2Ext*>(plugin)) {}
  virtual int getNbOutputs() const noexcept override;
  virtual nvinfer1::Dims getOutputDimensions(int index,
                                             const nvinfer1::Dims *inputDims,
                                             int nbInputs) noexcept override ;
  virtual void serialize(void* buffer) const noexcept override;
  virtual size_t getSerializationSize() const noexcept override;

  virtual int  initialize() noexcept override;
  virtual void terminate() noexcept override;

  virtual bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const noexcept override;
  virtual void configureWithFormat(const nvinfer1::Dims *inputDims, int nbInputs,
                                   const nvinfer1::Dims *outputDims, int nbOutputs,
                                   nvinfer1::DataType type,
                                   nvinfer1::PluginFormat format,
                                   int maxBatchSize) noexcept override;
  virtual size_t getWorkspaceSize(int maxBatchSize) const noexcept override;
  virtual int enqueue(int batchSize,
                      const void *const *inputs, void *const *outputs,
                      void *workspace, cudaStream_t stream) noexcept override;
};

// This makes a plugin compatible with onnx2trt::PluginFactory by serializing
// its plugin type.
class TypeSerializingPlugin : public PluginAdapter {
  UniqueOwnable _owned_plugin;
  Plugin* _plugin;
public:
  TypeSerializingPlugin(Plugin* plugin)
    : PluginAdapter(plugin), _owned_plugin(plugin), _plugin(plugin) {}
  void serialize(void* buffer) const noexcept override {
    const char* plugin_type = _plugin->getPluginType();
    serialize_value(&buffer, (const char*)REGISTERABLE_PLUGIN_MAGIC_STRING);
    serialize_value(&buffer, plugin_type);
    return _plugin->serialize(buffer);
  }
  size_t getSerializationSize() const noexcept override {
    const char* plugin_type = _plugin->getPluginType();
    // Note: +1 for NULL-terminated string
    return (sizeof(REGISTERABLE_PLUGIN_MAGIC_STRING) + 1 +
            strlen(plugin_type) +
            _plugin->getSerializationSize());
  }
  const char* getPluginType() const noexcept override {
    return _plugin->getPluginType();
  }
  void destroy() noexcept override { delete this; }
};

// Adapts nvinfer1::plugin::INvPlugin into onnx2trt::Plugin
// (This enables existing NV plugins to be used in this plugin infrastructure)
class NvPlugin : public PluginAdapter {
  nvinfer1::IPluginV2Ext*  _plugin;
public:
  NvPlugin(nvinfer1::IPluginV2Ext* plugin)
    : PluginAdapter(plugin), _plugin(plugin) {}
  virtual const char* getPluginType() const noexcept override;
  virtual void destroy() noexcept override;
};

} // namespace onnx2trt
