"""
解析讯飞录音文件转写返回的原始 JSON 结果。

用法:
    python result_parse.py <原始JSON文件路径> [--output 输出文件路径]

数据结构 (按讯飞文档):
    content.orderResult (JSON字符串) → lattice[] → json_1best (JSON字符串) → st
        st.bg / st.ed  : 句子开始/结束时间 (毫秒)
        st.rl          : 说话人角色编号 (正整数字符串, 开启角色分离才有)
        st.rt[].ws[]   : 词段集合
            ws.wb / ws.we : 词段开始/结束 (≈10ms, 相对于 bg)
            ws.cw[]       : 候选词
                cw.w  : 识别文字
                cw.wp : 词属性 (n=正常, p=标点, s=语气词, g=分隔符)
"""

import argparse
import json
import sys
from pathlib import Path


def parse_raw_result(raw_json: dict) -> list[dict]:
    """
    从讯飞原始 API 返回中解析出分段结果。

    Args:
        raw_json: 完整的 API 响应 dict (含 code, content 等)

    Returns:
        segments 列表, 每项: {"speaker": str, "begin": int(ms), "end": int(ms), "text": str}
    """
    order_result_str = raw_json["content"]["orderResult"]
    order_result = json.loads(order_result_str)

    lattice_list = order_result.get("lattice", [])
    segments = []

    for item in lattice_list:
        json_1best_str = item.get("json_1best", "{}")
        st = json.loads(json_1best_str).get("st", {})

        # bg / ed 单位: 毫秒
        begin_ms = int(st.get("bg", 0))
        end_ms = int(st.get("ed", 0))
        # rl: 角色编号
        speaker = st.get("rl", "")

        # 拼接该句所有词
        words = []
        for rt in st.get("rt", []):
            for ws in rt.get("ws", []):
                for cw in ws.get("cw", []):
                    w = cw.get("w", "")
                    wp = cw.get("wp", "n")
                    if w and wp != "g":  # g = 分隔符/空白, 跳过
                        words.append(w)

        text = "".join(words)
        if text.strip():
            segments.append({
                "speaker": f"说话人{speaker}" if speaker else "",
                "begin": begin_ms,
                "end": end_ms,
                "text": text,
            })

    return segments


def merge_segments(segments: list[dict]) -> list[dict]:
    """合并连续相同说话人的段落。"""
    if not segments:
        return []

    merged = [dict(segments[0])]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg["speaker"] == prev["speaker"]:
            prev["end"] = seg["end"]
            prev["text"] += seg["text"]
        else:
            merged.append(dict(seg))
    return merged


def ms_to_time(ms: int) -> str:
    """毫秒 → HH:MM:SS.mmm"""
    total_s, millis = divmod(ms, 1000)
    h, remainder = divmod(total_s, 3600)
    m, s = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{millis:03d}"


def format_transcript(segments: list[dict], *, merge: bool = True) -> str:
    """格式化为带时间戳和说话人的文字稿。"""
    items = merge_segments(segments) if merge else segments
    lines = []
    for seg in items:
        time_range = f"[{ms_to_time(seg['begin'])} - {ms_to_time(seg['end'])}]"
        speaker = f"【{seg['speaker']}】" if seg["speaker"] else ""
        lines.append(f"{time_range} {speaker}{seg['text']}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="解析讯飞转写原始 JSON 结果")
    parser.add_argument("file", help="原始 JSON 文件路径 (如 demo_5min_raw.json)")
    parser.add_argument("--output", "-o", help="输出文件路径 (默认: <输入文件名>_parsed.txt)")
    parser.add_argument("--no-merge", action="store_true", help="不合并连续相同说话人的段落")
    args = parser.parse_args()

    json_path = Path(args.file)
    if not json_path.is_file():
        print(f"错误: 文件不存在 - {json_path}", file=sys.stderr)
        sys.exit(1)

    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)

    segments = parse_raw_result(raw)
    transcript = format_transcript(segments, merge=not args.no_merge)

    # 输出路径
    output_path = Path(args.output) if args.output else json_path.with_name(f"{json_path.stem}_parsed.txt")
    output_path.write_text(transcript, encoding="utf-8")

    print(f"✅ 解析完成: {len(segments)} 个原始段落")
    print(f"   输出文件: {output_path}")
    print(f"\n--- 前 5 段预览 ---")
    for line in transcript.split("\n")[:5]:
        print(line)


if __name__ == "__main__":
    main()
