"""The DataForge design system as a Superset (Ant Design) theme.

Tokens are converted from the frontend's oklch palette (src/assets/styles.css):
Ivory for surfaces, Charcoal for text and primary controls, Vermilion as a signal only.
Controls are pills, content is square.
"""

IVORY_50 = "#FCFCF7"
IVORY_100 = "#F5F5EB"
IVORY_200 = "#EBEBD9"
IVORY_300 = "#DCDDC2"
IVORY_400 = "#C5C5A5"
VERMILION_50 = "#FBECE9"
VERMILION_300 = "#F48D7D"
VERMILION_500 = "#E95036"
VERMILION_700 = "#AF3A23"
VERMILION_800 = "#852917"
CHARCOAL_300 = "#919699"
CHARCOAL_400 = "#6B7175"
CHARCOAL_500 = "#52585B"
CHARCOAL_700 = "#373D40"
CHARCOAL_900 = "#1D2023"
SUCCESS = "#416548"
WARNING = "#82641B"
BORDER = "rgba(55, 61, 64, 0.14)"
LINE_STRONG = "rgba(55, 61, 64, 0.40)"

FONT_URL = (
    "https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,200..500"
    "&family=Instrument+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap"
)
FONT_SANS = '"Instrument Sans", system-ui, sans-serif'
FONT_MONO = '"JetBrains Mono", ui-monospace, monospace'

# Chart series: the same order as the frontend's --chart-1 … --chart-5, then lighter steps.
CHART_COLORS = [
    VERMILION_500,
    CHARCOAL_700,
    SUCCESS,
    WARNING,
    CHARCOAL_400,
    VERMILION_300,
    CHARCOAL_300,
    "#7E9A83",
    "#B89A55",
    IVORY_400,
]


def dataforge_theme(static_prefix: str = "/dataforge-static") -> dict:
    return {
        "algorithm": "default",
        "token": {
            # Brand
            "brandAppName": "DataForge",
            "brandLogoAlt": "DataForge",
            "brandLogoUrl": f"{static_prefix}/logo-primary.png",
            "brandLogoHref": "/superset/welcome/",
            "brandLogoHeight": "26px",
            "brandLogoMargin": "16px 0",
            "brandSpinnerUrl": f"{static_prefix}/logo-submark.png",
            "brandIconMaxWidth": 37,
            # Colour: charcoal drives controls, vermilion is kept for signals and errors.
            "colorPrimary": CHARCOAL_700,
            "colorLink": VERMILION_700,
            "colorLinkHover": VERMILION_500,
            "colorError": VERMILION_800,
            "colorWarning": WARNING,
            "colorSuccess": SUCCESS,
            "colorInfo": CHARCOAL_500,
            "colorTextBase": CHARCOAL_700,
            "colorText": CHARCOAL_700,
            "colorTextSecondary": CHARCOAL_500,
            "colorTextTertiary": CHARCOAL_400,
            "colorBgBase": IVORY_50,
            "colorBgLayout": IVORY_100,
            "colorBgContainer": IVORY_50,
            "colorBgElevated": IVORY_50,
            "colorBgSpotlight": CHARCOAL_700,
            "colorFillSecondary": IVORY_200,
            "colorFillTertiary": IVORY_200,
            "colorFillQuaternary": IVORY_100,
            "colorBorder": LINE_STRONG,
            "colorBorderSecondary": BORDER,
            "colorSplit": BORDER,
            "controlItemBgActive": IVORY_300,
            "controlItemBgHover": IVORY_200,
            "controlOutline": "rgba(55, 61, 64, 0.20)",
            "colorEditorSelection": VERMILION_50,
            # Type
            "fontUrls": [FONT_URL],
            "fontFamily": FONT_SANS,
            "fontFamilyCode": FONT_MONO,
            "fontSize": 14,
            "fontSizeXS": "8",
            "fontSizeXXL": "28",
            "fontWeightNormal": "400",
            "fontWeightLight": "300",
            "fontWeightStrong": "600",
            "fontWeightBold": "700",
            # Shape: content is square (small radius), controls are pills (see components).
            "borderRadius": 6,
            "borderRadiusLG": 6,
            "borderRadiusSM": 6,
            "borderRadiusXS": 4,
            "boxShadow": "none",
            "boxShadowSecondary": "0 8px 24px rgba(29, 32, 35, 0.10)",
            "transitionTiming": 0.2,
        },
        "components": {
            "Button": {"borderRadius": 999, "borderRadiusLG": 999, "borderRadiusSM": 999, "primaryShadow": "none"},
            "Input": {"borderRadius": 999},
            "Select": {"borderRadius": 999},
            "Tag": {"borderRadiusSM": 999},
            "Layout": {"headerBg": IVORY_200, "siderBg": IVORY_200, "bodyBg": IVORY_100},
            "Menu": {"itemBg": IVORY_200, "itemSelectedBg": IVORY_300, "itemSelectedColor": CHARCOAL_900},
            "Card": {"borderRadiusLG": 6},
            "Modal": {"borderRadiusLG": 6},
            "Table": {"headerBg": IVORY_200, "rowHoverBg": IVORY_100},
        },
    }


CATEGORICAL_SCHEME = {
    "id": "dataforge",
    "description": "DataForge chart colours",
    "label": "DataForge",
    "isDefault": True,
    "colors": CHART_COLORS,
}

SEQUENTIAL_SCHEME = {
    "id": "dataforgeIvoryToVermilion",
    "description": "DataForge sequential",
    "label": "DataForge ivory to vermilion",
    "isDefault": True,
    "isDiverging": False,
    "colors": [IVORY_100, VERMILION_50, VERMILION_300, VERMILION_500, VERMILION_700, VERMILION_800],
}
