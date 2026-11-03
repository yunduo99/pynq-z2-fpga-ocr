"""Optional full-frame RPC to the user's PC. Disabled unless backend='pc'."""
import json
import socket
import struct
import time

import cv2


def recv_exact(conn, size):
    data = bytearray()
    while len(data) < size:
        part = conn.recv(size-len(data))
        if not part:
            raise ConnectionError('PC OCR connection closed')
        data.extend(part)
    return bytes(data)


def recognize(terminal, gray):
    start = time.perf_counter()
    ok, encoded = cv2.imencode('.png', gray, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise RuntimeError('Cannot encode PC OCR frame')
    timeout = max(1, getattr(terminal, 'pc_timeout_ms', 15000)) / 1000
    with socket.create_connection((getattr(terminal, 'pc_host', '192.168.2.1'),
                                   int(getattr(terminal, 'pc_port', 9989))), timeout=timeout) as conn:
        conn.settimeout(timeout)
        conn.sendall(struct.pack('!III', gray.shape[1], gray.shape[0], encoded.size) + encoded.tobytes())
        size = struct.unpack('!I', recv_exact(conn, 4))[0]
        if size > 16*1024*1024:
            raise RuntimeError('PC OCR response too large')
        response = json.loads(recv_exact(conn, size).decode('utf-8'))
    if not response.get('ok') or response.get('protocol') != 'fullframe-v9':
        raise RuntimeError(response.get('error', 'PC server must support fullframe-v9'))
    lines = []
    for item in response.get('lines', []):
        x, y, w, h = [int(v) for v in item['rect']]
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x+w > gray.shape[1] or y+h > gray.shape[0]:
            raise RuntimeError('PC returned out-of-image text coordinates')
        lines.append({'text': str(item['text']), 'rect': (x, y, w, h),
                      'confidence': float(item['confidence'])})
    response['roundtrip_ms'] = (time.perf_counter()-start)*1000
    return lines, response
