import json
import subprocess
from logging import Logger
from typing import List, Optional

import xmltodict

from common.config_manager import config


class SystemSpecs:
    """
    Helpers for pulling system specs, such as cpu, memory, gpu, etc

    Note: these commands aren't supported on OSX, but they are on Ubuntu.
    """

    def __init__(self, logger: Logger):
        self.logger = logger

    def get_gpu_spec(self) -> Optional[List[dict]]:
        """
        :return: GPU specs,
        ex. `[{'model': 'NVIDIA GeForce RTX 3080 Laptop GPU', 'memory': '16384 MiB', 'pwr_limit': '80.00 W'}]`
        """
        try:
            r = subprocess.run(args=['nvidia-smi', '-x', '-q'], capture_output=True)
        # I know what you are thinking. "This can't be... A `FileNotFoundError`
        # for a missing command?" I'm here to assure you that that's true
        # `FileNotFoundError: [Errno 2] No such file or directory: 'nvidia-smi'`
        except FileNotFoundError:
            self.logger.error('`nvidia-smi` command not found')
            return None

        j = xmltodict.parse(r.stdout.decode('utf-8'))

        nvidia_smi_specs = j['nvidia_smi_log']['gpu']
        is_single_gpu = isinstance(j['nvidia_smi_log']['gpu'], dict)
        if is_single_gpu:
            nvidia_smi_specs = [j['nvidia_smi_log']['gpu']]

        gpu_specs = []
        for gpu_spec in nvidia_smi_specs:
            model = gpu_spec['product_name']
            memory = gpu_spec['fb_memory_usage']['total']
            pwr_limit = gpu_spec['gpu_power_readings']['default_power_limit']
            gpu_specs.append({'model': model, 'memory': memory, 'pwr_limit': pwr_limit})

        return gpu_specs

    def get_cpu_spec(self) -> Optional[dict]:
        """
        :return: CPU specs,
        ex. {'architecture': 'x86_64', 'cpus': '16', 'model_name': '11th Gen Intel(R) Core(TM) i9-11950H @ 2.60GHz', 'threads_per_core': '2'}
        """
        def clean_key(key: str) -> str:
            return key.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(':', '')

        try:
            r = subprocess.run(args=['lscpu', '-J'], capture_output=True)
        except FileNotFoundError:
            self.logger.error('`lscpu` command not found')
            return None

        j = json.loads(r.stdout.decode('utf-8'))
        o = {clean_key(x['field']): x for x in j.get('lscpu')}
        l = ('architecture', 'cpus', 'model_name', 'threads_per_core')
        return {x: o[x]['data'] for x in l}

    def get_mem_spec(self) -> Optional[str]:
        """
        :return: System memory in gigabytes, ex. `31.21 GiB`
        """
        try:
            r = subprocess.run(args=['grep', 'MemTotal', '/proc/meminfo'], capture_output=True)
        except FileNotFoundError:
            self.logger.error('`grep` command not found')
            return None
        # If `/proc/meminfo` isn't found

        err = r.stderr.decode('utf-8')
        if '/proc/meminfo' in err:
            self.logger.error('`/proc/meminfo` is not available in this system')
            return None

        s = r.stdout.decode('utf-8')
        v = s.replace('MemTotal:', '').replace('kB', '').strip()
        g = int(v)/1024/1024
        return f'{g:.2f} GiB'

    def get_specs(self) -> dict:
        """
        :return: Aggregate of all specs
        """
        return {
            'gpu': self.get_gpu_spec(),
            'cpu': self.get_cpu_spec(),
            'mem': self.get_mem_spec(),}
