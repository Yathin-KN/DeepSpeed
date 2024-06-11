import os

class PluginManager:
    def __init__(self):
        self.plugins = {}
        self.default_device_type = 'nvme'
        self.detect_plugins()

    def available_plugins(self):
        return list(self.plugins.keys())
    
    def detect_plugins(self):
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../csrc/aio/plugins'))
        for device_type in os.listdir(base_path):
            plugin_dir = os.path.join(base_path, device_type)
            if os.path.isdir(plugin_dir):
                self.plugins[device_type] = {
                    'include_paths': self.get_include_paths(device_type),
                    'source_paths': self.get_sources(device_type)
                }

    def get_sources(self, device_type):
        plugin_base_path = os.path.join('csrc/aio/plugins', device_type)
        common_base_path = 'csrc/aio/common'

        abs_plugin_base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../', plugin_base_path))
        abs_common_base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../', common_base_path))

        if not os.path.exists(abs_plugin_base_path):
            device_type = self.default_device_type
            plugin_base_path = os.path.join('csrc/aio/plugins', device_type)
            abs_plugin_base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../', plugin_base_path))

        plugin_sources = [os.path.join(plugin_base_path, f) for f in os.listdir(abs_plugin_base_path) if f.endswith(('.cpp', '.h'))]
        common_sources = [os.path.join(common_base_path, f) for f in os.listdir(abs_common_base_path) if f.endswith(('.cpp', '.h'))]
        
        return plugin_sources + common_sources

    def get_include_paths(self, device_type):
        plugin_base_path = os.path.join('csrc/aio/plugins', device_type)
        common_base_path = 'csrc/aio/common'
        
        abs_plugin_base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../', plugin_base_path))
        if not os.path.exists(abs_plugin_base_path):
            device_type = self.default_device_type
            plugin_base_path = os.path.join('csrc/aio/plugins', device_type)

        return [plugin_base_path, common_base_path]

    def get_plugin_info(self, device_type):
        if device_type in self.plugins:
            return self.plugins[device_type]
        else:
            return None
