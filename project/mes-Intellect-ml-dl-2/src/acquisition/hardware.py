import logging

logger = logging.getLogger(__name__)


class ProbeCardController:
    """探針卡自動化控制通訊介面 (VISA/SCPI 指令集模擬)"""

    def __init__(self, resource_string: str = "TCPIP0::192.168.1.100::inst0::INSTR"):
        self.resource_string = resource_string
        self.is_connected = False

    def connect_instrument(self):
        # 模擬 pyvisa.ResourceManager().open_resource()
        self.is_connected = True
        logger.info(
            f"Connected to Prober Tester via VISA Interface: {self.resource_string}"
        )

    def apply_probe_voltages(self, channels: int, voltage_range: list):
        if not self.is_connected:
            raise ConnectionError("儀器未連線")
        # 模擬發送 SCPI 標準通訊指令
        scpi_command = f":SOUR:VOLT:RANGE {voltage_range[1]}; :SENS:SWE:CHAN {channels}"
        logger.info(f"Sending SCPI Command to Probe Card: {scpi_command}")
        return True
