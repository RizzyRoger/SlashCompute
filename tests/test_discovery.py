"""A hosted pool advertises itself on the LAN so "Find on LAN" can join it."""

import logging

from fastapi.testclient import TestClient
from zeroconf import Zeroconf

from slashcompute.common.config import MDNS_SERVICE_TYPE, EngineConfig
from slashcompute.coordinator.app import create_app


def test_hosted_pool_advertises_on_the_lan(tmp_path, caplog):
    cfg = EngineConfig(home=tmp_path / "home", coordinator_port=18765)
    with caplog.at_level(logging.WARNING, logger="slashcompute.coordinator.app"):
        with TestClient(create_app(cfg, advertise=True)) as client:
            adv = client.app.state.advertiser
            assert adv is not None, caplog.text
            # Resolve this coordinator by name: another pool on the same LAN may also be advertising.
            zc = Zeroconf()
            try:
                info = zc.get_service_info(MDNS_SERVICE_TYPE, adv.name, timeout=3000)
            finally:
                zc.close()
    assert "mDNS advertising unavailable" not in caplog.text
    assert info is not None and info.port == 18765
    assert info.properties.get(b"path") == b"/ws/agent"
