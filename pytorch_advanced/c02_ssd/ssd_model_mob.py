"""モブプロでやったやつ.
IO を揃えて名前を一緒にすることで、後で組み込みやすくする

"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from serde import serde
from serde.json import from_json


def make_datapath_list_zumen(
    rootpath: str | Path,
) -> tuple[list[Path], list[Path], list[Path], list[Path]]:
    """
    データへのパスを格納したリストを作成する。

    Parameters
    ----------
    rootpath : str
        データフォルダへのパス

    Returns
    -------
    ret : train_img_list, train_anno_list, val_img_list, val_anno_list
        データへのパスを格納したリスト
    """
    rootpath = Path(rootpath)

    # 画像ファイルとアノテーションファイルへのパスのテンプレートを作成
    img_dir = rootpath / "data" / "images"
    annot_dir = rootpath / "data" / "annots" / "object-view"

    # 訓練と検証、それぞれのファイルのID（ファイル名）を取得する
    train_id_names = rootpath / "data" / "train_list.txt"
    val_id_names = rootpath / "data" / "val_list.txt"

    # 訓練データの画像ファイルとアノテーションファイルへのパスリストを作成
    train_ids = train_id_names.read_text().split("\n")
    val_ids = val_id_names.read_text().split("\n")
    train_img_list, train_anno_list = image_and_annotation_list(
        train_ids, img_dir, annot_dir)
    val_img_list, val_anno_list = image_and_annotation_list(
        val_ids, img_dir, annot_dir)

    return train_img_list, train_anno_list, val_img_list, val_anno_list


def image_and_annotation_list(
        file_ids: list[str], image_dir: Path,
        annot_dir: Path) -> tuple[list[Path], list[Path]]:
    return [image_dir / f"{f}.png"
            for f in file_ids], [annot_dir / f"{f}.json" for f in file_ids]


# AnocciデータをPascal VOC形式に変換する
@serde
@dataclass
class ArrowPattern:
    axis_length: float
    head_length: float
    head_degree: float


@serde
@dataclass
class ArrowPosition:
    head: Tuple[float, float]
    dir: Tuple[float, float]


@serde
@dataclass
class Arrow:
    pattern: ArrowPattern
    position: ArrowPosition


@serde
@dataclass
class Dimension:
    p1: float
    p2: float


@serde
@dataclass
class Triangle:
    foot: Tuple[float, float]
    head: Tuple[float, float]


@serde
@dataclass
class TriangleRow:
    triangles: List[Triangle]


@serde
@dataclass
class OldJisSurfaceRoughness:
    nose: Tuple[float, float]
    foot: Tuple[float, float]
    head: str
    ra: Optional[float]


@serde
@dataclass
class NewJisSurfaceRoughness:
    nose: Tuple[float, float]
    foot: Tuple[float, float]
    head: str
    ra: Optional[float]


@serde
@dataclass
class LineRange:
    lower: float
    upper: float


@serde
@dataclass
class Box2d:
    x: LineRange
    y: LineRange


@serde
@dataclass
class DimensionBox:
    box2d: Box2d
    box_type: str


@serde
@dataclass
class Anocci:
    image_dimensions: Tuple[int, int]
    arrows: List[Arrow]
    dimensions: List[Dimension]
    dimension_boxes: List[DimensionBox]
    triangle_rows: List[TriangleRow]
    old_jis_surface_roughnesses: List[OldJisSurfaceRoughness]
    new_jis_surface_roughnesses: List[NewJisSurfaceRoughness]
    object_views: List[Box2d]


def anocci_from_json(json_path: Path) -> Anocci:
    return from_json(Anocci, json_path.read_text())


def voc_from_anocci_json(json_path: Path,
                         normalize=True,
                         *args,
                         **kwargs) -> np.ndarray:
    """allow kwargs"""
    anocci = anocci_from_json(json_path)
    object_view_label = 0
    width, height = anocci.image_dimensions
    if normalize:
        object_views = list(
            map(
                lambda o: [
                    o.x.lower / width,
                    o.y.lower / height,
                    o.x.upper / width,
                    o.y.upper / height,
                    object_view_label,
                ],
                anocci.object_views,
            ))
    else:
        object_views = list(
            map(
                lambda o: [
                    o.x.lower,
                    o.y.lower,
                    o.x.upper,
                    o.y.upper,
                    object_view_label,
                ],
                anocci.object_views,
            ))

    return np.array(object_views, dtype=np.float32)


def test_something():
    print("hello")
    print('me')
    print('me')
