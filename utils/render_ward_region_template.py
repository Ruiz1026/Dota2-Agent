import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.widgets import Button, TextBox
from PIL import Image


def _attach_cursor_readout(fig: plt.Figure, ax: plt.Axes) -> None:
    label = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#ffffff",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "#111111",
            "edgecolor": "#555555",
            "alpha": 0.7,
        },
    )

    def on_move(event) -> None:
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            label.set_text("")
            fig.canvas.draw_idle()
            return

        label.set_text(f"x={event.xdata:.1f}, y={event.ydata:.1f}")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)


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


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_point_list(value) -> bool:
    return (
        isinstance(value, list)
        and value
        and all(
            isinstance(point, (list, tuple))
            and len(point) == 2
            and all(_is_number(item) for item in point)
            for point in value
        )
    )


def _format_number(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def _format_value(value, indent_level: int, indent: str = "  ") -> List[str]:
    current = indent * indent_level
    if isinstance(value, dict):
        if not value:
            return [current + "{}"]
        lines = [current + "{"]
        items = list(value.items())
        for idx, (key, val) in enumerate(items):
            val_lines = _format_value(val, indent_level + 1, indent)
            first = val_lines[0].lstrip()
            lines.append(
                indent * (indent_level + 1)
                + json.dumps(key, ensure_ascii=False)
                + ": "
                + first
            )
            lines.extend(val_lines[1:])
            if idx < len(items) - 1:
                lines[-1] += ","
        lines.append(current + "}")
        return lines
    if isinstance(value, list):
        if not value:
            return [current + "[]"]
        if _is_point_list(value):
            lines = [current + "["]
            for idx, point in enumerate(value):
                line = (
                    indent * (indent_level + 1)
                    + "["
                    + _format_number(point[0])
                    + ", "
                    + _format_number(point[1])
                    + "]"
                )
                if idx < len(value) - 1:
                    line += ","
                lines.append(line)
            lines.append(current + "]")
            return lines
        lines = [current + "["]
        for idx, item in enumerate(value):
            item_lines = _format_value(item, indent_level + 1, indent)
            lines.extend(item_lines)
            if idx < len(value) - 1:
                lines[-1] += ","
        lines.append(current + "]")
        return lines
    return [current + json.dumps(value, ensure_ascii=False)]


def _write_template_json(data: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    formatted = "\n".join(_format_value(data, 0)) + "\n"
    output_path.write_text(formatted, encoding="utf-8")


class RegionEditor:
    def __init__(
        self,
        template_path: Path,
        map_path: Path,
        output_path: Path,
        mode: str,
        key: Optional[str],
        label: Optional[str],
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
        self.default_key = key
        self.default_label = label
        self.precision = precision
        self.keep_open = keep_open

        self._load_template()

        self.fig, self.ax = plt.subplots(1, 1, figsize=(9, 9))
        img = Image.open(map_path)
        self.ax.imshow(img, extent=[64, 192, 64, 192], origin="upper")
        self.ax.set_xlim(64, 192)
        self.ax.set_ylim(64, 192)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title("点击添加锚点，回车生成新区域")

        self.region_artists: List[object] = []
        self._draw_existing_regions()
        self._attach_input_boxes()

        self.draft = RegionDraft(points=[])
        (self.preview_line,) = self.ax.plot([], [], color="#00d1b2", linewidth=1.6)
        self.preview_scatter = self.ax.scatter([], [], color="#00d1b2", s=20)

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def _attach_input_boxes(self) -> None:
        self.fig.subplots_adjust(bottom=0.28)
        button_y = 0.12
        button_h = 0.06
        button_w = 0.16
        clear_axes = self.fig.add_axes([0.08, button_y, button_w, button_h])
        save_axes = self.fig.add_axes([0.30, button_y, button_w, button_h])
        next_axes = self.fig.add_axes([0.52, button_y, button_w, button_h])
        refresh_axes = self.fig.add_axes([0.74, button_y, button_w, button_h])

        self.clear_button = Button(clear_axes, "清空")
        self.save_button = Button(save_axes, "保存")
        self.next_button = Button(next_axes, "下一条")
        self.refresh_button = Button(refresh_axes, "刷新")
        self.clear_button.on_clicked(self._on_clear_clicked)
        self.save_button.on_clicked(self._on_save_clicked)
        self.next_button.on_clicked(self._on_next_clicked)
        self.refresh_button.on_clicked(self._on_refresh_clicked)

        key_axes = self.fig.add_axes([0.12, 0.04, 0.32, 0.06])
        label_axes = self.fig.add_axes([0.56, 0.04, 0.32, 0.06])
        self.key_box = TextBox(key_axes, "Key", initial=self.default_key or "")
        self.label_box = TextBox(label_axes, "Label", initial=self.default_label or "")

    def _attach_help_text(self) -> None:
        help_lines = [
            "左键：添加锚点",
            "右键/Backspace：撤销上一个点",
            "Enter：生成区域并保存",
            "Esc：清空当前草稿",
            "按钮：清空/保存/下一条/刷新",
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
                        linewidth=1.2,
                        edgecolor="#f85149",
                        facecolor="#f85149",
                        alpha=0.3,
                        linestyle="solid",
                    )
                    self._track_region_artist(rect)
                elif area.get("type") == "polygon":
                    points = area.get("points") or []
                    if len(points) >= 3:
                        poly = Polygon(
                            points,
                            closed=True,
                            linewidth=1.2,
                            edgecolor="#f85149",
                            facecolor="#f85149",
                            alpha=0.3,
                            linestyle="solid",
                        )
                        self._track_region_artist(poly)

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
        if event.inaxes in (self.key_box.ax, self.label_box.ax):
            return
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

    def _clear_draft(self) -> None:
        self.draft.clear()
        self._refresh_preview()

    def _clear_inputs(self) -> None:
        self.key_box.set_val("")
        self.label_box.set_val("")

    def _on_clear_clicked(self, _event) -> None:
        self._clear_draft()

    def _on_save_clicked(self, _event) -> None:
        self._finalize_region()

    def _on_next_clicked(self, _event) -> None:
        if self._finalize_region(force_keep_open=True):
            self._clear_inputs()

    def _on_refresh_clicked(self, _event) -> None:
        self._refresh_from_disk()

    def _resolve_metadata(self) -> Tuple[str, str]:
        key = (self.key_box.text or "").strip()
        if not key:
            key = self.default_key or f"region_{len(self.regions) + 1}"
        label = (self.label_box.text or "").strip()
        if not label:
            label = self.default_label or key
        return key, label

    def _finalize_region(self, force_keep_open: Optional[bool] = None) -> bool:
        if self.mode == "bbox" and len(self.draft.points) < 2:
            print("需要两个点才能生成 bbox。")
            return False
        if self.mode == "polygon" and len(self.draft.points) < 3:
            print("需要至少三个点才能生成 polygon。")
            return False

        key, label = self._resolve_metadata()

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
        self.regions.append(region)

        self._draw_new_region(area)
        self._save_template()
        self.draft.clear()
        self._refresh_preview()

        if force_keep_open is None:
            should_close = not self.keep_open
        else:
            should_close = not force_keep_open
        if should_close:
            plt.close(self.fig)
        return True

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
            self._track_region_artist(rect)
        elif area["type"] == "polygon":
            poly = Polygon(
                area["points"],
                closed=True,
                linewidth=1.4,
                edgecolor="#00d1b2",
                facecolor="#00d1b2",
                alpha=0.18,
            )
            self._track_region_artist(poly)
        self.fig.canvas.draw_idle()

    def _track_region_artist(self, artist: object) -> None:
        self.ax.add_patch(artist)
        self.region_artists.append(artist)

    def _clear_region_artists(self) -> None:
        for artist in self.region_artists:
            try:
                artist.remove()
            except ValueError:
                continue
        self.region_artists.clear()

    def _load_template(self) -> None:
        with self.template_path.open("r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.regions = _ensure_regions(self.data)

    def _refresh_from_disk(self) -> None:
        self._load_template()
        self._clear_region_artists()
        self._draw_existing_regions()
        self._clear_draft()
        self.fig.canvas.draw_idle()
        print("已刷新并重新加载模板。")

    def _save_template(self) -> None:
        _write_template_json(self.data, self.output_path)
        print(f"已保存：{self.output_path}")

    def show(self) -> None:
        plt.show()


def render_template(template_path: Path, map_path: Path, output_path: Path) -> Path:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    with template_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    img = Image.open(map_path)
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    ax.imshow(img, extent=[64, 192, 64, 192], origin="upper")
    ax.set_title("Dota 2 区域模板（整合）")
    ax.set_xlim(64, 192)
    ax.set_ylim(64, 192)
    ax.set_xticks([])
    ax.set_yticks([])

    colors = [
        "#3fb950",
        "#f85149",
        "#ffd700",
        "#58a6ff",
        "#a371f7",
        "#f2cc60",
        "#7ee787",
        "#ffa657",
        "#d2a8ff",
        "#c9d1d9",
    ]
    side_styles = {
        "radiant": {"alpha": 0.18, "linestyle": "solid"},
        "dire": {"alpha": 0.12, "linestyle": "dashed"},
    }
    default_style = {"alpha": 0.15, "linestyle": "solid"}
    side_labels = {"radiant": "天辉", "dire": "夜魇"}

    regions = data.get("regions")
    if regions is None:
        regions = []
        for side in ("radiant", "dire"):
            for key, info in data.get(side, {}).items():
                merged = dict(info)
                merged.setdefault("key", key)
                merged.setdefault("side", side)
                regions.append(merged)

    color_by_key = {}
    for idx, info in enumerate(regions):
        key = info.get("key") or f"region_{idx + 1}"
        side = info.get("side")
        style = side_styles.get(side, default_style)
        if key not in color_by_key:
            color_by_key[key] = colors[len(color_by_key) % len(colors)]
        color = color_by_key[key]
        label_points = []
        for area in info.get("areas", []):
            if area.get("type") == "bbox":
                x0 = area["x_min"]
                y0 = area["y_min"]
                w = area["x_max"] - area["x_min"]
                h = area["y_max"] - area["y_min"]
                rect = Rectangle(
                    (x0, y0),
                    w,
                    h,
                    linewidth=1.2,
                    edgecolor=color,
                    facecolor=color,
                    alpha=style["alpha"],
                    linestyle=style["linestyle"],
                )
                ax.add_patch(rect)
                label_points.append((x0 + w / 2, y0 + h / 2))
            elif area.get("type") == "polygon":
                points = area.get("points") or []
                if len(points) >= 3:
                    poly = Polygon(
                        points,
                        closed=True,
                        linewidth=1.2,
                        edgecolor=color,
                        facecolor=color,
                        alpha=style["alpha"],
                        linestyle=style["linestyle"],
                    )
                    ax.add_patch(poly)
                    cx = sum(p[0] for p in points) / len(points)
                    cy = sum(p[1] for p in points) / len(points)
                    label_points.append((cx, cy))

        if label_points:
            cx = sum(p[0] for p in label_points) / len(label_points)
            cy = sum(p[1] for p in label_points) / len(label_points)
            label = info.get("label")
            if not label:
                prefix = side_labels.get(side, side or "区域")
                label = f"{prefix}-{key}"
            ax.text(
                cx,
                cy,
                label,
                color=color,
                fontsize=7,
                ha="center",
                va="center",
                weight="bold",
            )

    _attach_cursor_readout(fig, ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    return output_path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Render ward region template overlay")
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
        default=str(root / "ward_analysis" / "ward_region_template_740.png"),
        help="Path to output image",
    )
    parser.add_argument(
        "--edit",
        action="store_true",
        help="Open interactive editor to add region",
    )
    parser.add_argument(
        "--format-json",
        action="store_true",
        help="Reformat template JSON with fewer line breaks",
    )
    parser.add_argument(
        "--edit-out",
        default="",
        help="Output JSON path for editor (default: overwrite template)",
    )
    parser.add_argument(
        "--mode",
        choices=("polygon", "bbox"),
        default="polygon",
        help="Shape type to add when editing",
    )
    parser.add_argument("--key", default="", help="Region key (optional)")
    parser.add_argument("--label", default="", help="Region label (optional)")
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
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open interactive window",
    )
    args = parser.parse_args()

    if args.format_json:
        template_path = Path(args.template)
        data = json.loads(template_path.read_text(encoding="utf-8"))
        _write_template_json(data, template_path)
        print(f"已格式化：{template_path}")
    elif args.edit:
        output_path = Path(args.edit_out) if args.edit_out else Path(args.template)
        editor = RegionEditor(
            template_path=Path(args.template),
            map_path=Path(args.map),
            output_path=output_path,
            mode=args.mode,
            key=args.key or None,
            label=args.label or None,
            precision=args.precision,
            keep_open=args.keep_open,
        )
        editor.show()
    else:
        output_path = render_template(Path(args.template), Path(args.map), Path(args.out))
        print(output_path)
        if not args.no_show:
            plt.show()


if __name__ == "__main__":
    main()
