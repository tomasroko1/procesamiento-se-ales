from pathlib import Path

import numpy as np
import pytest

from hc3.io.parser import parse_xml
from hc3.schema import LFP, Session, Unit


def test_session_dataclass():
    s = Session(
        id="test",
        path=Path("/tmp"),
        n_channels=64,
        sampling_rate=20000,
        lfp_sampling_rate=1250,
        shank_map={1: [0, 1, 2]},
    )
    assert s.n_channels == 64
    assert s.path == Path("/tmp")


def test_unit_dataclass():
    u = Unit(
        id="u1",
        session_id="s1",
        shank_id=1,
        cluster_id=2,
        spike_times=np.array([0.1, 0.2, 0.3]),
    )
    assert u.n_spikes == 3
    assert u.firing_rate(duration=1.0) == 3.0
    assert u.firing_rate(duration=0.5) == 6.0


def test_lfp_dataclass():
    data = np.zeros((100, 4))
    lfp = LFP(data=data, sampling_rate=1000, channel_ids=[0, 1, 2, 3])
    assert lfp.data.shape == (100, 4)


def test_parse_xml_not_found():
    with pytest.raises(FileNotFoundError):
        parse_xml(Path("non_existent.xml"))
