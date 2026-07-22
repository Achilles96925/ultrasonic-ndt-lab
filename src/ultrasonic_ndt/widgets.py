"""Matplotlib 控件的共享工厂，保证各视图风格一致。"""

from matplotlib.markers import MarkerStyle
from matplotlib.widgets import CheckButtons


def checkbox_markers(checkbox: CheckButtons):
    """返回勾选标记集合，兼容 Matplotlib 3.11 前后的内部属性改名。"""
    return checkbox._buttons if hasattr(checkbox, "_buttons") else checkbox._checks


def make_circular_checkbox(
    axes,
    labels,
    actives,
    *,
    edgecolor: str = "#1565C0",
    check_color: str = "#1565C0",
) -> CheckButtons:
    """创建圆形（单选风格）的 CheckButtons。

    CheckButtons 默认用方形框和叉号标记，这里把框和标记都换成圆形，
    让"开关式"勾选框看起来像单选按钮，更接近真实仪器界面。
    """
    checkbox = CheckButtons(
        axes,
        labels,
        actives,
        frame_props={"edgecolor": edgecolor, "linewidth": 1.2},
        check_props={"color": check_color},
    )
    circle = MarkerStyle("o")
    circle_path = circle.get_path().transformed(circle.get_transform())
    paths = [circle_path] * len(labels)
    checkbox._frames.set_paths(paths)
    markers = checkbox_markers(checkbox)
    markers.set_paths(paths)
    markers.set_sizes(checkbox._frames.get_sizes() * 0.26)
    return checkbox
