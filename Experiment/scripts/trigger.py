import socket
import time
from datetime import datetime
import asyncio
import random

import numpy as np

UDP_IP = "10.0.0.1"
UDP_PORT = 26000

def send_trigger(trigger_id, img_id, exposure_time, exp_timestamp, ip=UDP_IP, port=UDP_PORT):
    async def async_trigger():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        trigger_timestamp = datetime.now().timestamp()
        sock.sendto(np.array([trigger_id, img_id, exposure_time, exp_timestamp, trigger_timestamp], dtype="double").tobytes(), (ip, port))

    asyncio.run(async_trigger())


def trigger_test(msg, ip=UDP_IP, port=UDP_PORT):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    trigger_timestamp = datetime.now().timestamp()
    sock.sendto(np.array([msg, trigger_timestamp], dtype="double").tobytes(), (ip, port))


if __name__ == '__main__':
    i = 0
    while True:
        i += 1
        trigger_test(i)
        time.sleep(random.random())
        print('send')
