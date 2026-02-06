import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from PIL import Image


@dataclass
class RegionDraft:
    points: List[Tuple[float, float]]

    def clear(self) -> None:
        self.points.clear()

    def add(self, x: float, y: float) -> None:
        self.points.append((x, y))

    def pop(self) -> None:
        if self.points:
            self.points.pop()


def _ensure_regions(data: dict) -> List[dict]:
    regions = data.get("regions")
    if regions is None:
        regions = []
        for side in ("radiant", "dire"):
            for key, info in data.get(side, {}).items():
                merged = dict(info)
                merged.setdefault("key", key)
                merged.setdefault("side", side)
                regions.append(merged)
        data["regions"] = regions
    return regions


def _round_value(value: float, precision: int) -> float:
    rounded = round(value, precision)
    if precision == 0:
        return int(round(rounded))
    return rounded


def _collect_metadata(
    key: Optional[str],
    label: Optional[str],
    side: Optional[str],
    default_index: int,
) -> Tuple[str, str, Optional[str]]:
    if not key:
        key = input(f"Region key (default region_{default_index}): ").strip()
        if not key:
            key = f"region_{default_index}"
    if not label:
        label = input(f"Region label (default {key}): ").strip()
        if not label:
            label = key
    if side is None:
        side_value = input("Side (radiant/dire/blank): ").strip().lower()
        side = side_value or None
    return key, label, side


class RegionEditor:
    def __init__(
        self,
        template_path: Path,
        map_path: Path,
        output_path: Path,
        mode: str,
        key: Optional[str],
        label: Optional[str],
        side: Optional[str],
        precision: int,
        keep_open: bool,
    ) -> None:
        plt.rcParams["font.sans-serif"] = [
            "Microsoft YaHei",
            "SimHei",
            "Arial Unicode MS",
            "DejaVu Sans",
        ]
        plt.rcParams["axes.unicode_minus"] = False

        self.template_path = template_path
        self.map_path = map_path
        self.output_path = output_path
        self.mode = mode
        self.key = key
        self.label = label
        self.side = side
        self.precision = precision
        self.keep_open = keep_open

        with template_path.open("r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.regions = _ensure_regions(self.data)

        self.fig, self.ax = plt.subplots(1, 1, figsize=(9, 9))
        img = Image.open(map_path)
        self.ax.imshow(img, extent=[64, 192, 64, 192], origin="upper")
        self.ax.set_xlim(64, 192)
        self.ax.set_ylim(64, 192)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title("点击添加锚点，回车生成新区域")

        self._draw_existing_regions()
        self._attach_help_text()

        self.draft = RegionDraft(points=[])
        (self.preview_line,) = self.ax.plot([], [], color="#00d1b2", linewidth=1.6)
        self.preview_scatter = self.ax.scatter([], [], color="#00d1b2", s=20)

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def _attach_help_text(self) -> None:
        help_lines = [
            "左键：添加锚点",
            "右键/Backspace：撤销上一个点",
            "Enter：生成区域并保存",
            "Esc：清空当前草稿",
        ]
        if self.mode == "bbox":
            help_lines.insert(0, "当前模式：bbox（点击两个点）")
        else:
            help_lines.insert(0, "当前模式：polygon（至少3点）")
        self.ax.text(
            0.01,
            0.01,
            "\n".join(help_lines),
            transform=self.ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            color="#ffffff",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "#111111",
                "edgecolor": "#555555",
                "alpha": 0.7,
            },
        )

    def _draw_existing_regions(self) -> None:
        for info in self.regions:
            for area in info.get("areas", []):
                if area.get("type") == "bbox":
                    rect = Rectangle(
                        (area["x_min"], area["y_min"]),
                        area["x_max"] - area["x_min"],
                        area["y_max"] - area["y_min"],
                        linewidth=0.8,
                        edgecolor="#c9d1d9",
                        facecolor="#c9d1d9",
                        alpha=0.08,
                        linestyle="dotted",
                    )
                    self.ax.add_patch(rect)
                elif area.get("type") == "polygon":
                    points = area.get("points") or []
                    if len(points) >= 3:
                        poly = Polygon(
                            points,
                            closed=True,
                            linewidth=0.8,
                            edgecolor="#c9d1d9",
                            facecolor="#c9d1d9",
                            alpha=0.08,
                            linestyle="dotted",
                        )
                        self.ax.add_patch(poly)

    def _refresh_preview(self) -> None:
        if not self.draft.points:
            self.preview_line.set_data([], [])
            self.preview_scatter.set_offsets([[float("nan"), float("nan")]])
        else:
            xs = [p[0] for p in self.draft.points]
            ys = [p[1] for p in self.draft.points]
            self.preview_line.set_data(xs, ys)
            self.preview_scatter.set_offsets(self.draft.points)
        self.fig.canvas.draw_idle()

    def on_click(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        if event.button == 1:
            self.draft.add(event.xdata, event.ydata)
            self._refresh_preview()
        elif event.button == 3:
            self.draft.pop()
            self._refresh_preview()

    def on_key(self, event) -> None:
        if event.key in ("backspace", "delete"):
            self.draft.pop()
            self._refresh_preview()
            return
        if event.key == "escape":
            self.draft.clear()
            self._refresh_preview()
            return
        if event.key == "enter":
            self._finalize_region()

    def _finalize_region(self) -> None:
        if self.mode == "bbox" and len(self.draft.points) < 2:
            print("需要两个点才能生成 bbox。")
            return
        if self.mode == "polygon" and len(self.draft.points) < 3:
            print("需要至少三个点才能生成 polygon。")
            return

        key, label, side = _collect_metadata(
            self.key,
            self.label,
            self.side,
            len(self.regions) + 1,
        )

        if self.mode == "bbox":
            (x1, y1), (x2, y2) = self.draft.points[:2]
            x_min = _round_value(min(x1, x2), self.precision)
            y_min = _round_value(min(y1, y2), self.precision)
            x_max = _round_value(max(x1, x2), self.precision)
            y_max = _round_value(max(y1, y2), self.precision)
            area = {"type": "bbox", "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}
        else:
            points = [
                [
                    _round_value(x, self.precision),
                    _round_value(y, self.precision),
                ]
                for x, y in self.draft.points
            ]
            area = {"type": "polygon", "points": points}

        region = {"key": key, "label": label, "areas": [area]}
        if side:
            region["side"] = side
        self.regions.append(region)

        self._draw_new_region(area)
        self._save_template()
        self.draft.clear()
        self._refresh_preview()

        if not self.keep_open:
            plt.close(self.fig)

    def _draw_new_region(self, area: dict) -> None:
        if area["type"] == "bbox":
            rect = Rectangle(
                (area["x_min"], area["y_min"]),
                area["x_max"] - area["x_min"],
                area["y_max"] - area["y_min"],
                linewidth=1.4,
                edgecolor="#00d1b2",
                facecolor="#00d1b2",
                alpha=0.18,
            )
            self.ax.add_patch(rect)
        elif area["type"] == "polygon":
            poly = Polygon(
                area["points"],
                closed=True,
                linewidth=1.4,
                edgecolor="#00d1b2",
                facecolor="#00d1b2",
                alpha=0.18,
            )
            self.ax.add_patch(poly)
        self.fig.canvas.draw_idle()

    def _save_template(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"已保存：{self.output_path}")

    def show(self) -> None:
        plt.show()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Interactively add region to template")
    parser.add_argument(
        "--template",
        default=str(root / "api_samples" / "ward_region_template_740.json"),
        help="Path to region template JSON",
    )
    parser.add_argument(
        "--map",
        default=str(root / "maps" / "740.jpeg"),
        help="Path to map image",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output JSON path (default: overwrite template)",
    )
    parser.add_argument(
        "--mode",
        choices=("polygon", "bbox"),
        default="polygon",
        help="Shape type to add",
    )
    parser.add_argument("--key", default="", help="Region key (optional)")
    parser.add_argument("--label", default="", help="Region label (optional)")
    parser.add_argument(
        "--side",
        default=None,
        help="Side tag (radiant/dire or leave blank)",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=1,
        help="Round coordinate precision",
    )
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="Keep window open after saving",
    )
    args = parser.parse_args()

    template_path = Path(args.template)
    output_path = Path(args.out) if args.out else template_path
    editor = RegionEditor(
        template_path=template_path,
        map_path=Path(args.map),
        output_path=output_path,
        mode=args.mode,
        key=args.key or None,
        label=args.label or None,
        side=args.side,
        precision=args.precision,
        keep_open=args.keep_open,
    )
    editor.show()


if __name__ == "__main__":
    main()
