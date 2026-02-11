"""PLY 스트리밍 로더 — numpy 기반 고속 로딩 + 랜덤 샘플링으로 대용량 포인트 클라우드를 메모리 효율적으로 로드."""

from collections.abc import Callable
from pathlib import Path

import numpy as np


# PLY 타입 → numpy dtype 매핑
_PLY_TO_NUMPY = {
    "double": "<f8", "float": "<f4", "float32": "<f4", "float64": "<f8",
    "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
    "ushort": "<u2", "uint16": "<u2", "short": "<i2", "int16": "<i2",
    "uint": "<u4", "uint32": "<u4", "int": "<i4", "int32": "<i4",
}


def read_ply_header(filepath: str | Path) -> dict:
    """PLY 헤더를 파싱하여 vertex 수와 속성 정보를 반환한다."""
    filepath = Path(filepath)
    properties = []
    vertex_count = 0
    header_size = 0

    with open(filepath, "rb") as f:
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            if line == "end_header":
                header_size = f.tell()
                break
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line.startswith("property"):
                parts = line.split()
                properties.append({"type": parts[1], "name": parts[2]})

    # numpy structured dtype 생성
    np_dtype = np.dtype(
        [(p["name"], _PLY_TO_NUMPY.get(p["type"], "<f4")) for p in properties]
    )

    return {
        "vertex_count": vertex_count,
        "properties": properties,
        "header_size": header_size,
        "vertex_size": np_dtype.itemsize,
        "np_dtype": np_dtype,
        "filepath": filepath,
    }


def load_ply_sampled(
    filepath: str | Path,
    max_points: int = 5_000_000,
    progress_callback: Callable | None = None,
    seed: int = 42,
    chunk_size: int = 1_000_000,
) -> dict:
    """대용량 PLY 파일을 numpy 청크 읽기 + 랜덤 샘플링으로 로드한다.

    사전에 샘플 인덱스를 생성한 뒤, 청크 단위로 numpy structured array를 읽으며
    해당 인덱스의 포인트만 추출한다. Python 루프 없이 C-speed로 동작.

    Args:
        filepath: PLY 파일 경로
        max_points: 최대 샘플링 포인트 수
        progress_callback: 진행률 콜백 함수 (current, total) -> None

    Returns:
        dict with keys: points (N,3), colors (N,3), intensity (N,), classification (N,),
                        total_vertices (int), sampled_vertices (int)
    """
    header = read_ply_header(filepath)
    total = header["vertex_count"]
    dt = header["np_dtype"]
    prop_names = [p["name"] for p in header["properties"]]

    sample_size = min(max_points, total)

    # 사전에 샘플 인덱스를 정렬된 상태로 생성 (순차 접근 보장)
    rng = np.random.default_rng(seed=seed)
    if sample_size < total:
        sample_indices = np.sort(rng.choice(total, size=sample_size, replace=False))
    else:
        sample_indices = np.arange(total)

    # 결과 배열 사전 할당
    points = np.empty((sample_size, 3), dtype=np.float32)
    has_color = "red" in prop_names
    has_intensity = "scalar_Intensity" in prop_names
    has_classification = "scalar_Classification" in prop_names
    colors = np.empty((sample_size, 3), dtype=np.float32) if has_color else None
    intensity = np.empty(sample_size, dtype=np.float32) if has_intensity else None
    classification = np.empty(sample_size, dtype=np.float32) if has_classification else None

    # 청크 단위 스트리밍 읽기 (chunk_size vertices/chunk ≈ 35MB/chunk)
    filled = 0

    with open(header["filepath"], "rb") as f:
        f.seek(header["header_size"])

        for chunk_start in range(0, total, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total)

            # 이 청크에 해당하는 샘플 인덱스 찾기
            mask = (sample_indices >= chunk_start) & (sample_indices < chunk_end)
            local_indices = sample_indices[mask] - chunk_start

            if len(local_indices) == 0:
                # 샘플이 없는 청크는 건너뛰기 (seek)
                f.seek(dt.itemsize * (chunk_end - chunk_start), 1)
            else:
                # numpy로 청크 전체를 한번에 읽기 (C-speed)
                chunk = np.frombuffer(
                    f.read(dt.itemsize * (chunk_end - chunk_start)), dtype=dt
                )

                # 샘플 인덱스만 추출
                sampled = chunk[local_indices]
                n = len(sampled)

                points[filled : filled + n, 0] = sampled["x"].astype(np.float32)
                points[filled : filled + n, 1] = sampled["y"].astype(np.float32)
                points[filled : filled + n, 2] = sampled["z"].astype(np.float32)

                if colors is not None:
                    colors[filled : filled + n, 0] = sampled["red"].astype(np.float32) / 255.0
                    colors[filled : filled + n, 1] = sampled["green"].astype(np.float32) / 255.0
                    colors[filled : filled + n, 2] = sampled["blue"].astype(np.float32) / 255.0

                if intensity is not None:
                    intensity[filled : filled + n] = sampled["scalar_Intensity"]

                if classification is not None:
                    classification[filled : filled + n] = sampled["scalar_Classification"]

                filled += n

            if progress_callback:
                progress_callback(chunk_end, total)

    return {
        "points": points[:filled],
        "colors": colors[:filled] if colors is not None else None,
        "intensity": intensity[:filled] if intensity is not None else None,
        "classification": classification[:filled] if classification is not None else None,
        "total_vertices": total,
        "sampled_vertices": filled,
    }
