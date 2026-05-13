"""Tests for plotting PNSN-style phase pick dictionaries on waveforms."""

from __future__ import annotations

import numpy as np
import pytest

from seismo_code.toolkit import plot_stream


def test_plot_stream_accepts_relative_picks_without_obspy_objects(tmp_path):
    class Stats:
        network = "X1"
        station = "53085"
        location = "01"
        channel = "BHZ"
        starttime = 0.0
        sampling_rate = 100.0

    class Trace:
        stats = Stats()
        id = "X1.53085.01.BHZ"

        def __init__(self):
            self.data = np.sin(np.linspace(0, 20, 1000)).astype(np.float32)

        def times(self):
            return np.arange(self.data.size) / self.stats.sampling_rate

    picks = [
        {
            "station": "X1.53085.01",
            "channel": "BHZ",
            "phase": "P",
            "time_rel_s": 3.0,
        }
    ]
    outfile = tmp_path / "matplotlib_relative_picks.png"

    result = plot_stream([Trace()], title="Relative picks", outfile=str(outfile), picks=picks)

    assert result == str(outfile)
    assert outfile.exists()
    assert outfile.stat().st_size > 0


def test_plot_stream_accepts_pnsn_pick_time_fields_and_station_labels(tmp_path):
    obspy = pytest.importorskip("obspy")

    start = obspy.UTCDateTime("2012-07-27T00:00:00")
    stream = obspy.Stream()
    for channel in ("BHZ", "BHN", "BHE"):
        trace = obspy.Trace(data=np.sin(np.linspace(0, 20, 1000)).astype(np.float32))
        trace.stats.network = "X1"
        trace.stats.station = "53085"
        trace.stats.location = "01"
        trace.stats.channel = channel
        trace.stats.starttime = start
        trace.stats.sampling_rate = 100.0
        stream += trace

    picks = [
        {
            "station": "X1.53085.01",
            "channel": "BHZ",
            "phase": "P",
            "time_abs": "2012-07-27T00:00:03.000000Z",
            "time_rel_s": 3.0,
        },
        {
            "station": "X1.53085.01",
            "phase": "S",
            "time_rel_s": 6.0,
        },
    ]
    outfile = tmp_path / "pnsn_picks_on_waveform.png"

    result = plot_stream(stream, title="PNSN picks", outfile=str(outfile), picks=picks)

    assert result == str(outfile)
    assert outfile.exists()
    assert outfile.stat().st_size > 0
