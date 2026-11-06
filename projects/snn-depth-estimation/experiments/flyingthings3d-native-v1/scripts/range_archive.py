"""Exact bounded HTTP ranges over immutable concatenated uncompressed TAR parts."""
import bisect
import json
import time
import requests


class RangeArchive:
    def __init__(self, config):
        self.config = config
        self.parts = config['parts']
        self.starts = [0]
        for part in self.parts:
            self.starts.append(self.starts[-1] + part['size'])
        self.session = requests.Session()
        self.urls = {}
        self.bytes_read = 0
        self.requests = 0

    def read(self, start, size):
        if start < 0 or size < 0 or start + size > self.starts[-1]:
            raise ValueError('Range outside archive')
        chunks = []
        while size:
            index = bisect.bisect_right(self.starts, start) - 1
            local = start - self.starts[index]
            count = min(size, self.parts[index]['size'] - local)
            end = local + count - 1
            for attempt in range(5):
                try:
                    url = self.urls.get(index, self.parts[index]['url'])
                    with self.session.get(url, headers={'Range': f'bytes={local}-{end}'}, stream=True, timeout=(15, 60)) as response:
                        self.requests += 1
                        expected = f'bytes {local}-{end}/{self.parts[index]["size"]}'
                        if response.status_code != 206 or response.headers.get('Content-Range') != expected:
                            self.urls.pop(index, None)
                            raise RuntimeError(f'Range not honored: HTTP{response.status_code}, {response.headers.get("Content-Range")}')
                        payload = response.raw.read(count + 1)
                        if len(payload) != count:
                            raise RuntimeError('Range payload length differs')
                        self.urls[index] = response.url
                        self.bytes_read += len(payload)
                    chunks.append(payload)
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(min(10, 2**attempt))
            start += count
            size -= count
        return b''.join(chunks)


def tar_header(payload, offset):
    if len(payload) != 512:
        raise ValueError('TAR header length')
    if not any(payload):
        return None
    stored = int(payload[148:156].strip(b'\0 ') or b'0', 8)
    if sum(payload[:148]) + 8*32 + sum(payload[156:]) != stored:
        raise ValueError('TAR header checksum at ' + str(offset))
    name = payload[:100].split(b'\0',1)[0].decode()
    prefix = payload[345:500].split(b'\0',1)[0].decode()
    if prefix:
        name = prefix + '/' + name
    size = int(payload[124:136].strip(b'\0 ') or b'0',8)
    return dict(path=name, size=size, header_offset=offset, data_offset=offset+512,
                type=payload[156:157].decode() or '0', next_offset=offset+512+((size+511)//512)*512)


def load(path):
    with open(path) as stream:
        return json.load(stream)
