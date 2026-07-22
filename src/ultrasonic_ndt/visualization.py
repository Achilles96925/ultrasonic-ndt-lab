"""成像示例共用的显示设置。"""

from matplotlib.colors import LinearSegmentedColormap


def white_blue_yellow_red_colormap() -> LinearSegmentedColormap:
    """返回低幅白色、高幅红色的 NDT 显示色带。"""
    return LinearSegmentedColormap.from_list(
        "white_blue_yellow_red",
        (
            (0.00, "#FFFFFF"),
            (0.12, "#FFFFFF"),
            (0.34, "#0757B9"),
            (0.68, "#FFE34C"),
            (1.00, "#D7191C"),
        ),
    )
