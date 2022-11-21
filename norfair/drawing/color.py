import re
from typing import Any, Hashable, Iterable, Tuple, Union

# types

ColorType = Tuple[int, int, int]
ColorLike = Union[ColorType, str]


def hex_to_bgr(hex_value: str) -> ColorType:
    """Converts conventional 6 digits hex colors to BGR tuples

    Parameters
    ----------
    hex_value : str
        hex value with leading `#` for instance `"#ff0000"`

    Returns
    -------
    Tuple[int, int, int]
        BGR values

    Raises
    ------
    ValueError
        if the string is invalid
    """
    if re.match("#[a-f0-9]{6}$", hex_value):
        return (
            int(hex_value[5:7], 16),
            int(hex_value[3:5], 16),
            int(hex_value[1:3], 16),
        )

    if re.match("#[a-f0-9]{3}$", hex_value):
        return (
            int(hex_value[3] * 2, 16),
            int(hex_value[2] * 2, 16),
            int(hex_value[1] * 2, 16),
        )
    raise ValueError(f"'{hex_value}' is not a valid color")


class Color:
    """
    Contains predefined colors.

    Colors are defined as a Tuple of integers between 0 and 255 expressing the values in BGR
    This is the format opencv uses.
    """

    # from PIL.ImageColors.colormap
    aliceblue = hex_to_bgr("#f0f8ff")
    antiquewhite = hex_to_bgr("#faebd7")
    aqua = hex_to_bgr("#00ffff")
    aquamarine = hex_to_bgr("#7fffd4")
    azure = hex_to_bgr("#f0ffff")
    beige = hex_to_bgr("#f5f5dc")
    bisque = hex_to_bgr("#ffe4c4")
    black = hex_to_bgr("#000000")
    blanchedalmond = hex_to_bgr("#ffebcd")
    blue = hex_to_bgr("#0000ff")
    blueviolet = hex_to_bgr("#8a2be2")
    brown = hex_to_bgr("#a52a2a")
    burlywood = hex_to_bgr("#deb887")
    cadetblue = hex_to_bgr("#5f9ea0")
    chartreuse = hex_to_bgr("#7fff00")
    chocolate = hex_to_bgr("#d2691e")
    coral = hex_to_bgr("#ff7f50")
    cornflowerblue = hex_to_bgr("#6495ed")
    cornsilk = hex_to_bgr("#fff8dc")
    crimson = hex_to_bgr("#dc143c")
    cyan = hex_to_bgr("#00ffff")
    darkblue = hex_to_bgr("#00008b")
    darkcyan = hex_to_bgr("#008b8b")
    darkgoldenrod = hex_to_bgr("#b8860b")
    darkgray = hex_to_bgr("#a9a9a9")
    darkgrey = hex_to_bgr("#a9a9a9")
    darkgreen = hex_to_bgr("#006400")
    darkkhaki = hex_to_bgr("#bdb76b")
    darkmagenta = hex_to_bgr("#8b008b")
    darkolivegreen = hex_to_bgr("#556b2f")
    darkorange = hex_to_bgr("#ff8c00")
    darkorchid = hex_to_bgr("#9932cc")
    darkred = hex_to_bgr("#8b0000")
    darksalmon = hex_to_bgr("#e9967a")
    darkseagreen = hex_to_bgr("#8fbc8f")
    darkslateblue = hex_to_bgr("#483d8b")
    darkslategray = hex_to_bgr("#2f4f4f")
    darkslategrey = hex_to_bgr("#2f4f4f")
    darkturquoise = hex_to_bgr("#00ced1")
    darkviolet = hex_to_bgr("#9400d3")
    deeppink = hex_to_bgr("#ff1493")
    deepskyblue = hex_to_bgr("#00bfff")
    dimgray = hex_to_bgr("#696969")
    dimgrey = hex_to_bgr("#696969")
    dodgerblue = hex_to_bgr("#1e90ff")
    firebrick = hex_to_bgr("#b22222")
    floralwhite = hex_to_bgr("#fffaf0")
    forestgreen = hex_to_bgr("#228b22")
    fuchsia = hex_to_bgr("#ff00ff")
    gainsboro = hex_to_bgr("#dcdcdc")
    ghostwhite = hex_to_bgr("#f8f8ff")
    gold = hex_to_bgr("#ffd700")
    goldenrod = hex_to_bgr("#daa520")
    gray = hex_to_bgr("#808080")
    grey = hex_to_bgr("#808080")
    green = (0, 128, 0)
    greenyellow = hex_to_bgr("#adff2f")
    honeydew = hex_to_bgr("#f0fff0")
    hotpink = hex_to_bgr("#ff69b4")
    indianred = hex_to_bgr("#cd5c5c")
    indigo = hex_to_bgr("#4b0082")
    ivory = hex_to_bgr("#fffff0")
    khaki = hex_to_bgr("#f0e68c")
    lavender = hex_to_bgr("#e6e6fa")
    lavenderblush = hex_to_bgr("#fff0f5")
    lawngreen = hex_to_bgr("#7cfc00")
    lemonchiffon = hex_to_bgr("#fffacd")
    lightblue = hex_to_bgr("#add8e6")
    lightcoral = hex_to_bgr("#f08080")
    lightcyan = hex_to_bgr("#e0ffff")
    lightgoldenrodyellow = hex_to_bgr("#fafad2")
    lightgreen = hex_to_bgr("#90ee90")
    lightgray = hex_to_bgr("#d3d3d3")
    lightgrey = hex_to_bgr("#d3d3d3")
    lightpink = hex_to_bgr("#ffb6c1")
    lightsalmon = hex_to_bgr("#ffa07a")
    lightseagreen = hex_to_bgr("#20b2aa")
    lightskyblue = hex_to_bgr("#87cefa")
    lightslategray = hex_to_bgr("#778899")
    lightslategrey = hex_to_bgr("#778899")
    lightsteelblue = hex_to_bgr("#b0c4de")
    lightyellow = hex_to_bgr("#ffffe0")
    lime = hex_to_bgr("#00ff00")
    limegreen = hex_to_bgr("#32cd32")
    linen = hex_to_bgr("#faf0e6")
    magenta = hex_to_bgr("#ff00ff")
    maroon = hex_to_bgr("#800000")
    mediumaquamarine = hex_to_bgr("#66cdaa")
    mediumblue = hex_to_bgr("#0000cd")
    mediumorchid = hex_to_bgr("#ba55d3")
    mediumpurple = hex_to_bgr("#9370db")
    mediumseagreen = hex_to_bgr("#3cb371")
    mediumslateblue = hex_to_bgr("#7b68ee")
    mediumspringgreen = hex_to_bgr("#00fa9a")
    mediumturquoise = hex_to_bgr("#48d1cc")
    mediumvioletred = hex_to_bgr("#c71585")
    midnightblue = hex_to_bgr("#191970")
    mintcream = hex_to_bgr("#f5fffa")
    mistyrose = hex_to_bgr("#ffe4e1")
    moccasin = hex_to_bgr("#ffe4b5")
    navajowhite = hex_to_bgr("#ffdead")
    navy = hex_to_bgr("#000080")
    oldlace = hex_to_bgr("#fdf5e6")
    olive = hex_to_bgr("#808000")
    olivedrab = hex_to_bgr("#6b8e23")
    orange = hex_to_bgr("#ffa500")
    orangered = hex_to_bgr("#ff4500")
    orchid = hex_to_bgr("#da70d6")
    palegoldenrod = hex_to_bgr("#eee8aa")
    palegreen = hex_to_bgr("#98fb98")
    paleturquoise = hex_to_bgr("#afeeee")
    palevioletred = hex_to_bgr("#db7093")
    papayawhip = hex_to_bgr("#ffefd5")
    peachpuff = hex_to_bgr("#ffdab9")
    peru = hex_to_bgr("#cd853f")
    pink = hex_to_bgr("#ffc0cb")
    plum = hex_to_bgr("#dda0dd")
    powderblue = hex_to_bgr("#b0e0e6")
    purple = hex_to_bgr("#800080")
    rebeccapurple = hex_to_bgr("#663399")
    red = hex_to_bgr("#ff0000")
    rosybrown = hex_to_bgr("#bc8f8f")
    royalblue = hex_to_bgr("#4169e1")
    saddlebrown = hex_to_bgr("#8b4513")
    salmon = hex_to_bgr("#fa8072")
    sandybrown = hex_to_bgr("#f4a460")
    seagreen = hex_to_bgr("#2e8b57")
    seashell = hex_to_bgr("#fff5ee")
    sienna = hex_to_bgr("#a0522d")
    silver = hex_to_bgr("#c0c0c0")
    skyblue = hex_to_bgr("#87ceeb")
    slateblue = hex_to_bgr("#6a5acd")
    slategray = hex_to_bgr("#708090")
    slategrey = hex_to_bgr("#708090")
    snow = hex_to_bgr("#fffafa")
    springgreen = hex_to_bgr("#00ff7f")
    steelblue = hex_to_bgr("#4682b4")
    tan = hex_to_bgr("#d2b48c")
    teal = hex_to_bgr("#008080")
    thistle = hex_to_bgr("#d8bfd8")
    tomato = hex_to_bgr("#ff6347")
    turquoise = hex_to_bgr("#40e0d0")
    violet = hex_to_bgr("#ee82ee")
    wheat = hex_to_bgr("#f5deb3")
    white = hex_to_bgr("#ffffff")
    whitesmoke = hex_to_bgr("#f5f5f5")
    yellow = hex_to_bgr("#ffff00")
    yellowgreen = hex_to_bgr("#9acd32")

    # seaborn tab20 colors
    tab1 = hex_to_bgr("#1f77b4")
    tab2 = hex_to_bgr("#aec7e8")
    tab3 = hex_to_bgr("#ff7f0e")
    tab4 = hex_to_bgr("#ffbb78")
    tab5 = hex_to_bgr("#2ca02c")
    tab6 = hex_to_bgr("#98df8a")
    tab7 = hex_to_bgr("#d62728")
    tab8 = hex_to_bgr("#ff9896")
    tab9 = hex_to_bgr("#9467bd")
    tab10 = hex_to_bgr("#c5b0d5")
    tab11 = hex_to_bgr("#8c564b")
    tab12 = hex_to_bgr("#c49c94")
    tab13 = hex_to_bgr("#e377c2")
    tab14 = hex_to_bgr("#f7b6d2")
    tab15 = hex_to_bgr("#7f7f7f")
    tab16 = hex_to_bgr("#c7c7c7")
    tab17 = hex_to_bgr("#bcbd22")
    tab18 = hex_to_bgr("#dbdb8d")
    tab19 = hex_to_bgr("#17becf")
    tab20 = hex_to_bgr("#9edae5")
    # seaborn colorblind
    cb1 = hex_to_bgr("#0173b2")
    cb2 = hex_to_bgr("#de8f05")
    cb3 = hex_to_bgr("#029e73")
    cb4 = hex_to_bgr("#d55e00")
    cb5 = hex_to_bgr("#cc78bc")
    cb6 = hex_to_bgr("#ca9161")
    cb7 = hex_to_bgr("#fbafe4")
    cb8 = hex_to_bgr("#949494")
    cb9 = hex_to_bgr("#ece133")
    cb10 = hex_to_bgr("#56b4e9")


def parse_color(color_like: ColorLike) -> ColorType:
    """Makes best effort to parse the given value to a Color

    Parameters
    ----------
    color_like : ColorLike
        Can be one of:

        1. a string with the 6 digits hex value (`"#ff0000"`)
        2. a string with one of the names defined in Colors (`"red"`)
        3. a BGR tuple (`(0, 0, 255)`)

    Returns
    -------
    Color
        The BGR tuple.
    """
    if isinstance(color_like, str):
        if color_like.startswith("#"):
            return hex_to_bgr(color_like)
        else:
            return getattr(Color, color_like)
    # TODO: validate?
    return tuple([int(v) for v in color_like])


PALETTES = {
    "tab10": [
        Color.tab1,
        Color.tab3,
        Color.tab5,
        Color.tab7,
        Color.tab9,
        Color.tab11,
        Color.tab13,
        Color.tab15,
        Color.tab17,
        Color.tab19,
    ],
    "tab20": [
        Color.tab1,
        Color.tab2,
        Color.tab3,
        Color.tab4,
        Color.tab5,
        Color.tab6,
        Color.tab7,
        Color.tab8,
        Color.tab9,
        Color.tab10,
        Color.tab11,
        Color.tab12,
        Color.tab13,
        Color.tab14,
        Color.tab15,
        Color.tab16,
        Color.tab17,
        Color.tab18,
        Color.tab19,
        Color.tab20,
    ],
    "colorblind": [
        Color.cb1,
        Color.cb2,
        Color.cb3,
        Color.cb4,
        Color.cb5,
        Color.cb6,
        Color.cb7,
        Color.cb8,
        Color.cb9,
        Color.cb10,
    ],
}


class Palette:
    """
    Class to control the color pallete for drawing.

    Examples
    --------
    Change palette:
    >>> from norfair import Palette
    >>> Palette.set("colorblind")
    >>> # or a custom palette
    >>> from norfair import Color
    >>> Palette.set([Color.red, Color.blue, "#ffeeff"])
    """

    _colors = PALETTES["tab10"]
    _default_color = Color.black

    @classmethod
    def set(cls, palette: Union[str, Iterable[ColorLike]]):
        """
        Selects a color palette.

        Parameters
        ----------
        palette : Union[str, Iterable[ColorLike]]
            can be either
            - the name of one of the predefined palettes `tab10`, `tab20`, or `colorblind`
            - a list of ColorLike objects that can be parsed by [`parse_color`][norfair.drawing.color.parse_color]
        """
        if isinstance(palette, str):
            try:
                cls._colors = PALETTES[palette]
            except KeyError as e:
                raise ValueError(
                    f"Invalid palette name '{palette}', valid values are {PALETTES.keys()}"
                ) from e
        else:
            colors = []
            for c in palette:
                colors.append(parse_color(c))

            cls._colors = colors

    @classmethod
    def set_default_color(cls, color: ColorLike):
        """
        Selects the default color of `choose_color` when hashable is None.

        Parameters
        ----------
        color : ColorLike
            The new default color.
        """
        cls._default_color = parse_color(color)

    @classmethod
    def choose_color(cls, hashable: Hashable) -> ColorType:
        if hashable is None:
            return cls._default_color
        return cls._colors[abs(hash(hashable)) % len(cls._colors)]
