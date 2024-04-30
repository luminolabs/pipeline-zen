import json
import os
import subprocess

from common.utils import get_root_path


class SystemSpecs:
    """
    Static methods for pulling system specs, such as cpu, memory, gpu, etc
    """

    @staticmethod
    def get_gpu_spec() -> dict:
        """
        :return: GPU specs,
        ex. `{'model': 'NVIDIA GeForce RTX 3080 Laptop GPU', 'memory': '16384 MiB', 'pwr_limit': '80.00 W'}`
        """
        r = subprocess.run(args=[os.path.join(get_root_path(), 'scripts', 'gpu_specs.sh')],
                           capture_output=True)
        return json.loads(r.stdout.decode('utf-8'))

    @staticmethod
    def get_cpu_spec() -> dict:
        """
        :return: CPU specs,
        ex. {'architecture': 'x86_64', 'cpus': '16', 'model_name': '11th Gen Intel(R) Core(TM) i9-11950H @ 2.60GHz', 'threads_per_core': '2'}
        """
        def clean_key(key: str) -> str:
            return key.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(':', '')

        r = subprocess.run(args=['lscpu', '-J'], capture_output=True)
        j = json.loads(r.stdout.decode('utf-8'))
        o = {clean_key(x['field']): x for x in j.get('lscpu')}
        l = ('architecture', 'cpus', 'model_name', 'threads_per_core')
        return {x: o[x]['data'] for x in l}

    @staticmethod
    def get_mem_spec() -> str:
        """
        :return: System memory in gigabytes, ex. `31.21 GiB`
        """
        r = subprocess.run(args=['grep', 'MemTotal', '/proc/meminfo'], capture_output=True)
        s = r.stdout.decode('utf-8')
        v = s.replace('MemTotal:', '').replace('kB', '').strip()
        g = int(v)/1024/1024
        return f'{g:.2f} GiB'

    @staticmethod
    def get_specs() -> dict:
        """
        :return: Aggregate of all specs
        """
        return {
            'gpu': SystemSpecs.get_gpu_spec(),
            'cpu': SystemSpecs.get_cpu_spec(),
            'mem': SystemSpecs.get_mem_spec(),}
