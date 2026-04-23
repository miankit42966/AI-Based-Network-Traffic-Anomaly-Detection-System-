from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


TCP_FLAG_MAP = {
    "F": 0x01,
    "S": 0x02,
    "R": 0x04,
    "P": 0x08,
    "A": 0x10,
    "U": 0x20,
}


@dataclass(slots=True)
class FlowFeatures:
    values: dict[str, float]


def _flags_to_numeric(flags: Any) -> int:
    if flags is None:
        return 0
    if isinstance(flags, (int, float)):
        return int(flags)

    encoded = 0
    for flag in str(flags):
        encoded |= TCP_FLAG_MAP.get(flag.upper(), 0)
    return encoded


def extract_flow_features(packet: dict[str, Any]) -> FlowFeatures:
    size = float(packet.get("size", 0.0))
    duration = float(packet.get("duration", 0.0))
    ttl = float(packet.get("ttl", 0.0))
    packets = max(float(packet.get("packets", 1.0)), 1.0)
    bytes_count = float(packet.get("bytes", size))

    values = {
        "Flow Duration": duration,
        "Flow Bytes/s": bytes_count / max(duration, 1.0),
        "Flow Packets/s": packets / max(duration, 1.0),
        "Packet Length Mean": size / packets,
        "Packet Length Std": float(packet.get("packet_length_std", 0.0)),
        "TTL": ttl,
        "Protocol": float(packet.get("protocol", 0.0)),
        "Source Port": float(packet.get("src_port", 0.0)),
        "Destination Port": float(packet.get("dst_port", 0.0)),
        "TCP Flags": float(_flags_to_numeric(packet.get("flags"))),
        "Inter Arrival Mean": float(packet.get("iat_mean", 0.0)),
    }
    return FlowFeatures(values=values)


def extract_from_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    engineered = frame.copy()

    if {"Total Fwd Packets", "Total Backward Packets"}.issubset(engineered.columns):
        total_packets = engineered["Total Fwd Packets"].fillna(0) + engineered["Total Backward Packets"].fillna(0)
        engineered["Flow Packets/s"] = total_packets / engineered["Flow Duration"].replace(0, 1)

    if {"Total Length of Fwd Packets", "Total Length of Bwd Packets"}.issubset(engineered.columns):
        total_bytes = engineered["Total Length of Fwd Packets"].fillna(0) + engineered["Total Length of Bwd Packets"].fillna(0)
        engineered["Flow Bytes/s"] = total_bytes / engineered["Flow Duration"].replace(0, 1)

    return engineered


def vectorize_features(feature_map: dict[str, float], expected_columns: list[str]) -> np.ndarray:
    row = [feature_map.get(column, 0.0) for column in expected_columns]
    return np.asarray(row, dtype=float).reshape(1, -1)
