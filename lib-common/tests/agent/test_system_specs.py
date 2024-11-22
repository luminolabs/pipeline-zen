import json
from unittest.mock import MagicMock, patch

import pytest

from common.agent.system_specs import SystemSpecsAgent

# Sample command outputs
NVIDIA_SMI_OUTPUT = """<?xml version="1.0" ?>
<nvidia_smi_log>
    <gpu>
        <product_name>NVIDIA GeForce RTX 3080 Laptop GPU</product_name>
        <fb_memory_usage>
            <total>16384 MiB</total>
        </fb_memory_usage>
        <gpu_power_readings>
            <default_power_limit>80.00 W</default_power_limit>
        </gpu_power_readings>
    </gpu>
</nvidia_smi_log>"""

NVIDIA_SMI_MULTI_GPU_OUTPUT = """<?xml version="1.0" ?>
<nvidia_smi_log>
    <gpu>
        <product_name>NVIDIA A100-SXM4-80GB</product_name>
        <fb_memory_usage>
            <total>81920 MiB</total>
        </fb_memory_usage>
        <gpu_power_readings>
            <default_power_limit>400.00 W</default_power_limit>
        </gpu_power_readings>
    </gpu>
    <gpu>
        <product_name>NVIDIA A100-SXM4-80GB</product_name>
        <fb_memory_usage>
            <total>81920 MiB</total>
        </fb_memory_usage>
        <gpu_power_readings>
            <default_power_limit>400.00 W</default_power_limit>
        </gpu_power_readings>
    </gpu>
</nvidia_smi_log>"""

LSCPU_OUTPUT = """{
    "lscpu": [
        {"field": "Architecture", "data": "x86_64"},
        {"field": "CPU(s)", "data": "16"},
        {"field": "Model name", "data": "11th Gen Intel(R) Core(TM) i9-11950H @ 2.60GHz"},
        {"field": "Thread(s) per core", "data": "2"}
    ]
}"""

MEMINFO_OUTPUT = "MemTotal:       32595424 kB"


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def system_specs_agent(mock_logger):
    return SystemSpecsAgent(mock_logger)


def test_get_gpu_spec_single(system_specs_agent):
    with patch('subprocess.run') as mock_run:
        mock_run.return_value.stdout = NVIDIA_SMI_OUTPUT.encode('utf-8')
        gpu_specs = system_specs_agent.get_gpu_spec()

        assert gpu_specs is not None
        assert len(gpu_specs) == 1
        assert gpu_specs[0]['model'] == 'NVIDIA GeForce RTX 3080 Laptop GPU'
        assert gpu_specs[0]['memory'] == '16384 MiB'
        assert gpu_specs[0]['pwr_limit'] == '80.00 W'


def test_get_gpu_spec_multiple(system_specs_agent):
    with patch('subprocess.run') as mock_run:
        mock_run.return_value.stdout = NVIDIA_SMI_MULTI_GPU_OUTPUT.encode('utf-8')
        gpu_specs = system_specs_agent.get_gpu_spec()

        assert gpu_specs is not None
        assert len(gpu_specs) == 2
        for gpu in gpu_specs:
            assert gpu['model'] == 'NVIDIA A100-SXM4-80GB'
            assert gpu['memory'] == '81920 MiB'
            assert gpu['pwr_limit'] == '400.00 W'


def test_get_gpu_spec_command_not_found(system_specs_agent, mock_logger):
    with patch('subprocess.run') as mock_run:
        mock_run.side_effect = FileNotFoundError()
        gpu_specs = system_specs_agent.get_gpu_spec()

        assert gpu_specs is None
        mock_logger.error.assert_called_once_with('`nvidia-smi` command not found')


def test_get_gpu_spec_invalid_output(system_specs_agent, mock_logger):
    with patch('subprocess.run') as mock_run:
        mock_run.return_value.stdout = b'Invalid XML output'
        gpu_specs = system_specs_agent.get_gpu_spec()

        assert gpu_specs is None
        mock_logger.error.assert_called_once()


def test_get_cpu_spec(system_specs_agent):
    with patch('subprocess.run') as mock_run:
        mock_run.return_value.stdout = LSCPU_OUTPUT.encode('utf-8')
        cpu_specs = system_specs_agent.get_cpu_spec()

        assert cpu_specs is not None
        assert cpu_specs['architecture'] == 'x86_64'
        assert cpu_specs['cpus'] == '16'
        assert cpu_specs['model_name'] == '11th Gen Intel(R) Core(TM) i9-11950H @ 2.60GHz'
        assert cpu_specs['threads_per_core'] == '2'


def test_get_cpu_spec_command_not_found(system_specs_agent, mock_logger):
    with patch('subprocess.run') as mock_run:
        mock_run.side_effect = FileNotFoundError()
        cpu_specs = system_specs_agent.get_cpu_spec()

        assert cpu_specs is None
        mock_logger.error.assert_called_once_with('`lscpu` command not found')


def test_get_cpu_spec_invalid_output(system_specs_agent, mock_logger):
    with patch('subprocess.run') as mock_run:
        mock_run.return_value.stdout = b'Invalid JSON output'
        cpu_specs = system_specs_agent.get_cpu_spec()

        assert cpu_specs is None
        mock_logger.error.assert_called_once()


def test_get_mem_spec(system_specs_agent):
    with patch('subprocess.run') as mock_run:
        mock_run.return_value.stdout = MEMINFO_OUTPUT.encode('utf-8')
        mem_specs = system_specs_agent.get_mem_spec()

        assert mem_specs is not None
        assert mem_specs == '31.09 GiB'


def test_get_mem_spec_command_not_found(system_specs_agent, mock_logger):
    with patch('subprocess.run') as mock_run:
        mock_run.side_effect = FileNotFoundError()
        mem_specs = system_specs_agent.get_mem_spec()

        assert mem_specs is None
        mock_logger.error.assert_called_once_with('`grep` command not found')


def test_get_mem_spec_file_not_found(system_specs_agent, mock_logger):
    with patch('subprocess.run') as mock_run:
        mock_run.return_value.stderr = b'/proc/meminfo: No such file or directory'
        mem_specs = system_specs_agent.get_mem_spec()

        assert mem_specs is None
        mock_logger.error.assert_called_once_with('`/proc/meminfo` is not available in this system')


def test_get_specs(system_specs_agent):
    with patch.multiple(SystemSpecsAgent,
                        get_gpu_spec=MagicMock(return_value=['gpu1']),
                        get_cpu_spec=MagicMock(return_value={'cpu': 'spec'}),
                        get_mem_spec=MagicMock(return_value='32GB')):
        specs = system_specs_agent.get_specs()

        assert specs == {
            'gpu': ['gpu1'],
            'cpu': {'cpu': 'spec'},
            'mem': '32GB'
        }


def test_get_specs_partial_failure(system_specs_agent):
    with patch.multiple(SystemSpecsAgent,
                        get_gpu_spec=MagicMock(return_value=None),
                        get_cpu_spec=MagicMock(return_value={'cpu': 'spec'}),
                        get_mem_spec=MagicMock(return_value='32GB')):
        specs = system_specs_agent.get_specs()

        assert specs == {
            'gpu': None,
            'cpu': {'cpu': 'spec'},
            'mem': '32GB'
        }