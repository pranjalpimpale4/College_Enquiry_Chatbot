from kivy.core.text import LabelBase

fonts_path = "assets/fonts/"

fonts = [
    {
        "name": "Lexend",
        "fn_regular": fonts_path + "Lexend-Regular.ttf",
        "fn_bold": fonts_path + "Lexend-Bold.ttf",
    },
    {
        "name": "LexendThin",
        "fn_regular": fonts_path + "Lexend-Thin.ttf",
    },
    {
        "name": "LexendLight",
        "fn_regular": fonts_path + "Lexend-Light.ttf",
    },
    {
        "name": "LexendMedium",
        "fn_regular": fonts_path + "Lexend-Medium.ttf",
    },
    {
        "name": "NotoSans",
        "fn_regular": fonts_path + "NotoSans-Regular.ttf",
    },
{
        "name": "Poppins",
        "fn_regular": fonts_path + "Poppins-Regular.ttf",
    },
    {
        "name": "Icons",
        "fn_regular": fonts_path + "Feather.ttf",
    },
    {
        "name": "DV_TT",
        "fn_regular": fonts_path + "DV-TTSurekhEN-Normal.ttf",
    },
    {
        "name": "Dev",
        "fn_regular": fonts_path + "Devanagari.ttf",
    },
{
        "name": "JB_Mono",
        "fn_regular": fonts_path + "JetBrainsMono-VariableFont_wght.ttf",
    },
]


def register_fonts():
    for font in fonts:
        LabelBase.register(**font)