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
CASES = (
    {"id": "zh", "reference": "zh", "seed": 37001, "target": "重庆的银行今天九点开门，请到二楼找林老师。"},
    {"id": "en", "reference": "en", "seed": 37002,
     "target": "When Neil arrived at the harbor, he checked the clock and called his sister."},
    {"id": "mixed", "reference": "zh", "seed": 37003,
     "target": "林老师说，please check the blue folder，我们下午三点见。"},
)
