"""Shared synthetic references and bilingual evaluation prompts."""
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "example_workflows" / "jr-benchmark-references"
REFERENCES = {
    "zh": ("zh.wav", "bc8e7c44dd794a3678e3253a2bda51fcb63e486605750996cfff19e45e5c9bd0",
           "清晨，我打开窗户，听见院子里的鸟儿在唱歌。街道还很安静，微风轻轻地吹过树梢。",
           "Microsoft Huihui Desktop"),
    "en": ("en.wav", "a3e6f39634747a64440b087da364c0a2132566dbf996846ae3025a63bfc7a5f7",
           "Early in the morning, I opened the window and heard birds singing in the garden. "
           "The air was cool and the street was quiet.", "Microsoft David Desktop"),
}
FIVE_LANG_FIXTURES = Path(__file__).resolve().parent / "resources/five-lang-voices"
FIVE_LANG_REFERENCES = {
    "ja": ("ja.wav", "a765d92518c0f7f615bdca46e8e8910b640e02bbb7d8484d57e4325178b315ab",
           "おはようございます。今日は風が気持ちいいですね。", "Drama 3 five-language synthetic voice"),
    "ko": ("ko.wav", "54612bba3caa8dcd2224bb1490d53d0150b5b653e9233f3ed0957a0b073874fd",
           "좋은 아침이에요. 창문을 열면 시원한 바람이 들어와요.", "Drama 3 five-language synthetic voice"),
    "yue": ("yue.wav", "d04c27f5f79d9ec9d548dbc52163cf87e01efab0f39302f1de52aaeb43e5c64e",
            "佢話今晚返嚟食飯，叫我哋唔使等。", "Drama 3 five-language synthetic voice"),
}
CASES = (
    {"id": "zh", "reference": "zh", "seed": 37001, "target": "重庆的银行今天九点开门，请到二楼找林老师。"},
    {"id": "en", "reference": "en", "seed": 37002,
     "target": "When Neil arrived at the harbor, he checked the clock and called his sister."},
    {"id": "mixed", "reference": "zh", "seed": 37003,
     "target": "林老师说，please check the blue folder，我们下午三点见。"},
)
