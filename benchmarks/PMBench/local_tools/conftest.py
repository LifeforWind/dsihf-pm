"""Offline regression suite uses dummy routing, never personal credentials."""
import socket
import pytest

@pytest.fixture(autouse=True)
def offline_test_environment(monkeypatch):
    def denied(*args,**kwargs):
        raise AssertionError('Regression tests may not open network connections')
    monkeypatch.setattr(socket.socket,'connect',denied)
    import memory_transport
    monkeypatch.setattr(memory_transport,'relay_config',lambda:('https://user-relay.example/v1','offline-test-key'))
