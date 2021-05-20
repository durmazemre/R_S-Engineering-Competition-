"""
Retrieve trace data from a SCPI compatible entity, like a T&M instrument or software.
"""
import socket
from typing import Union


class Instrument:
    def __init__(self, hostname: str, port: int = 5025, timeout: int = 10):
        """
        Very simple instrument connection class using a raw socket for communication

        :param hostname: hostname of IP address of the device or software to connect to
        :param port: optional, the TCP/IP port to connect to
        :param timeout: optional, the socket timeout in seconds
        """
        self.__socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.__socket.connect((hostname, port))
        self.__socket.settimeout(timeout)
        self.__buffer = 10 * 1024 * 1024

    def write(self, cmd: str):
        """
        Write an ASCII encoded string to the socket.

        :param cmd: command, string with ASCII chars
        """
        self.__socket.sendall(cmd.encode("ASCII") + b"\n")

    def read(self) -> str:
        """
        Eead an ASCII encoded string from the socket.

        All data is read from the socket buffer. This also means:
        No protocol awareness, no error checking, no separation in case of multiple simultaneous answers.

        :return: ASCII string
        """
        data = self.__socket.recv(self.__buffer)
        return data.decode("ASCII").rstrip("\n")

    def query(self, cmd: str) -> str:
        self.write(cmd)
        return self.read()
    
    def get_trace(self, window: int, trace: Union[int, str]) -> str:
        """
        Request trace data from a certain window.
        Window must be a window number (id),
        Trace may be a named trace (str) or a trace number (int).

        :param window: window id to read trace from
        :param trace: name or number of trace to read
        :return: trace data as string
        """
        if isinstance(trace, int):
            trace = f"TRACE{trace}"
        self.write(f"TRACE{window}:DATA? {trace}")
        return self.read()
