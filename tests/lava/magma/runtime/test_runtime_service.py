import logging
import os
import random
import unittest
from multiprocessing.managers import SharedMemoryManager

import numpy as np

from lava.magma.compiler.channels.pypychannel import PyPyChannel
from lava.magma.compiler.executable import Executable
from lava.magma.core.resources import HeadNode
from lava.magma.compiler.node import Node, NodeConfig
from lava.magma.core.decorator import implements
from lava.magma.core.model.nc.model import NcProcessModel
from lava.magma.core.model.py.model import AbstractPyProcessModel
from lava.magma.core.process.message_interface_enum import ActorType
from lava.magma.core.process.ports.ports import OutPort, InPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.sync.protocol import AbstractSyncProtocol
from lava.magma.core.sync.protocols.nxsdk_protocol import NxsdkProtocol
from lava.magma.runtime.runtime import Runtime
from lava.magma.runtime.runtime_services.enums import LoihiVersion
from lava.magma.runtime.runtime_services.runtime_service import (
    PyRuntimeService,
    NxSDKRuntimeService
)


class MockInterface:
    def __init__(self, smm):
        self.smm = smm


def create_channel(smm: SharedMemoryManager, name: str):
    mock = MockInterface(smm=smm)
    return PyPyChannel(
        mock,
        name,
        name,
        (1,),
        np.int32,
        8,
    )


class SimpleSyncProtocol(AbstractSyncProtocol):
    pass


class SimpleProcess(AbstractProcess):
    pass


@implements(proc=SimpleProcess, protocol=SimpleSyncProtocol)
class SimpleProcessModel(AbstractPyProcessModel):
    def run(self):
        pass


class SimplePyRuntimeService(PyRuntimeService):
    def run(self):
        pass


class TestRuntimeService(unittest.TestCase):
    def test_runtime_service_construction(self):
        sp = SimpleSyncProtocol()
        rs = SimplePyRuntimeService(protocol=sp)
        self.assertEqual(rs.protocol, sp)
        self.assertEqual(rs.service_to_runtime, None)
        self.assertEqual(rs.service_to_process, [])
        self.assertEqual(rs.runtime_to_service, None)
        self.assertEqual(rs.process_to_service, [])

    def test_runtime_service_start_run(self):
        pm = SimpleProcessModel(proc_params={})
        sp = SimpleSyncProtocol()
        rs = SimplePyRuntimeService(protocol=sp)
        smm = SharedMemoryManager()
        smm.start()
        runtime_to_service = create_channel(smm, name="runtime_to_service")
        service_to_runtime = create_channel(smm, name="service_to_runtime")
        service_to_process = [create_channel(smm, name="service_to_process")]
        process_to_service = [create_channel(smm, name="process_to_service")]
        runtime_to_service.dst_port.start()
        service_to_runtime.src_port.start()

        pm.service_to_process = service_to_process[0].dst_port
        pm.process_to_service = process_to_service[0].src_port
        pm.py_ports = []
        pm.start()
        rs.runtime_to_service = runtime_to_service.src_port
        rs.service_to_runtime = service_to_runtime.dst_port
        rs.service_to_process = [service_to_process[0].src_port]
        rs.process_to_service = [process_to_service[0].dst_port]
        rs.join()
        pm.join()
        smm.shutdown()


class NxSDKTestProcess(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(loglevel=logging.DEBUG, **kwargs)
        self.out = OutPort(shape=(2,))


@implements(proc=NxSDKTestProcess, protocol=NxsdkProtocol)
class NxSDKTestProcessModel(NcProcessModel):
    def start(self, num_steps):
        self.service_to_process.start()
        self.process_to_service.start()
        for p in self.nc_ports:
            p.start()
        self.board.run(num_steps)

    def stop(self):
        self.board.stop()

    def test_setup(self):
        self.nxCore = self.board.nxChips[0].nxCores[0]
        self.axon_map = self.nxCore.axonMap

    def test_idx(self, test_case: unittest.TestCase):
        value = random.getrandbits(15)
        # Setting the value of idx as value
        self.axon_map[0].idx = value
        self.axon_map.push(0)
        # Checking the value of idx
        self.axon_map.fetch(0)
        test_case.assertEqual(self.axon_map[0].idx, value)

        value = random.getrandbits(15)
        # Setting the value of idx as value
        self.axon_map[1].idx = value
        self.axon_map.push(1)
        # Checking the value of idx
        self.axon_map.fetch(1)
        test_case.assertEqual(self.axon_map[1].idx, value)

    def test_len(self, test_case: unittest.TestCase):
        value = random.getrandbits(13)
        # Setting the value of len as value
        self.axon_map[0].len = value
        self.axon_map.push(0)
        # Checking the value of len
        self.axon_map.fetch(0)
        test_case.assertEqual(self.axon_map[0].len, value)

        value = random.getrandbits(13)
        # Setting the value of len as value
        self.axon_map[1].len = value
        self.axon_map.push(1)
        # Checking the value of len
        self.axon_map.fetch(1)
        test_case.assertEqual(self.axon_map[1].len, value)

    def test_data(self, test_case: unittest.TestCase):
        value = random.getrandbits(36)
        # Setting the value of data as value
        self.axon_map[0].data = value
        self.axon_map.push(0)
        # Checking the value of data
        self.axon_map.fetch(0)
        test_case.assertEqual(self.axon_map[0].data, value)

        value = random.getrandbits(36)
        # Setting the value of data as value
        self.axon_map[1].data = value
        self.axon_map.push(1)
        # Checking the value of data
        self.axon_map.fetch(1)
        test_case.assertEqual(self.axon_map[1].data, value)


class TestNxSDKRuntimeService(unittest.TestCase):
    os.environ['SLURM'] = '1'
    os.environ['PARTITION'] = 'oheogulch'
    os.environ['LOIHI_GEN'] = 'N3B3'
    # os.environ['BOARD'] = 'ncl-og-02'
    # os.environ['LOIHI_GEN'] = 'N3B3'
    os.environ['BOARD'] = 'ncl-og-05'
    # os.environ['BOARD'] = 'ncl-og-02'
    # ## See nxsdk/arch/n3b/graph/nodesets/tests/test_axon_map.py
    # ## See nxsdk/driver/tests/test_read_write_opt.py

    def test_runtime_service_construction(self):
        p = NxsdkProtocol()
        rs = NxSDKRuntimeService(protocol=p)
        self.assertEqual(rs.protocol, p)
        self.assertEqual(rs.service_to_runtime, None)
        self.assertEqual(rs.service_to_process, [])
        self.assertEqual(rs.runtime_to_service, None)
        self.assertEqual(rs.process_to_service, [])

    def test_runtime_service_loihi_start_run(self):
        p = NxsdkProtocol()
        rs = NxSDKRuntimeService(protocol=p)
        pm = NxSDKTestProcessModel(proc_params={},
                                   board=rs.get_board())

        smm = SharedMemoryManager()
        smm.start()
        runtime_to_service = create_channel(smm, name="runtime_to_service")
        service_to_runtime = create_channel(smm, name="service_to_runtime")
        service_to_process = [create_channel(smm, name="service_to_process")]
        process_to_service = [create_channel(smm, name="process_to_service")]
        runtime_to_service.dst_port.start()
        service_to_runtime.src_port.start()

        pm.service_to_process = service_to_process[0].dst_port
        pm.process_to_service = process_to_service[0].src_port
        pm.nc_ports = []

        rs.num_steps = 1
        pm.start(rs.num_steps)

        rs.runtime_to_service = runtime_to_service.src_port
        rs.service_to_runtime = service_to_runtime.dst_port
        rs.service_to_process = [service_to_process[0].src_port]
        rs.process_to_service = [process_to_service[0].dst_port]

        rs.join()

        pm.test_setup()
        pm.test_idx(self)
        pm.test_len(self)
        pm.test_data(self)

        pm.join()
        pm.stop()
        smm.shutdown()

    @unittest.skip("runtime_to_runtime_service_to_process communication")
    def test_runtime_service_loihi_channels_start_run(self):
        node: Node = Node(HeadNode, [])
        exec: Executable = Executable()
        exec.node_configs.append(NodeConfig([node]))
        runtime: Runtime = Runtime(exec, ActorType.MultiProcessing)
        runtime.initialize()
        p = NxsdkProtocol()
        rs = NxSDKRuntimeService(protocol=p)
        pm = NcProcessModel(proc_params={},
                            board=rs.get_board())

        smm = SharedMemoryManager()
        smm.start()
        runtime_to_service = create_channel(smm, name="runtime_to_service")
        service_to_runtime = create_channel(smm, name="service_to_runtime")
        service_to_process = [create_channel(smm, name="service_to_process")]
        process_to_service = [create_channel(smm, name="process_to_service")]
        runtime_to_service.dst_port.start()
        service_to_runtime.src_port.start()

        rs.num_steps = 1

        rs.runtime_to_service = runtime_to_service.src_port
        rs.service_to_runtime = service_to_runtime.dst_port
        rs.service_to_process = [service_to_process[0].src_port]
        rs.process_to_service = [process_to_service[0].dst_port]

        pm.service_to_process = service_to_process[0].dst_port
        pm.process_to_service = process_to_service[0].src_port
        pm.nc_ports = []

        pm.start()

        rs.join()

        pm.stop()
        smm.shutdown()


if __name__ == '__main__':
    unittest.main()
