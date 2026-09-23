import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from seedplane import cli
from seedplane.llama_backend import MAGIC, RESP, Worker


class SecurityDefaultsTests(unittest.TestCase):
    def test_python_worker_defaults_to_loopback(self):
        with patch('sys.argv', ['seedplane', 'serve', '--bundle', 'unused']), patch.object(cli, 'cmd_serve') as serve:
            cli.main()
        self.assertEqual(serve.call_args.args[0].host, '127.0.0.1')

    def test_external_bind_requires_explicit_key(self):
        args = SimpleNamespace(host='0.0.0.0', threads=1, bundle='unused', device='cpu', port=52000)
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(SystemExit):
            cli.cmd_serve(args)


class NativeResponseTests(unittest.TestCase):
    def test_rejects_bad_magic(self):
        worker = Worker.__new__(Worker)
        worker.name = 'test'
        worker._read = lambda _: RESP.pack(0, 0.0, 0, -1, 0.0)
        with self.assertRaises(ConnectionError):
            worker.recv()

    def test_accepts_valid_response(self):
        worker = Worker.__new__(Worker)
        worker.name = 'test'
        worker._read = lambda _: RESP.pack(MAGIC, 1.5, 2, 3, 4.0)
        self.assertEqual(worker.recv()[:3], (1.5, 2, 3))


if __name__ == '__main__':
    unittest.main()
