"""
讯飞录音文件转写（标准版）
API 文档: https://www.xfyun.cn/doc/asr/ifasr_new/API.html

用法:
    python transcribe.py <音频文件路径> [--output 输出文件路径]
"""

import argparse
import base64
import hashlib
import hmac
import json
import logging
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
UPLOAD_URL = "https://raasr.xfyun.cn/v2/api/upload"
GET_RESULT_URL = "https://raasr.xfyun.cn/v2/api/getResult"

# 订单状态
STATUS_CREATED = 0
STATUS_PROCESSING = 3
STATUS_COMPLETED = 4
STATUS_FAILED = -1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 签名
# ---------------------------------------------------------------------------
def generate_signa(app_id: str, secret_key: str, ts: int) -> str:
    """
    signa = Base64(HmacSHA1(MD5(appid + ts), secretkey))
    """
    base_string = f"{app_id}{ts}"
    md5_hash = hashlib.md5(base_string.encode("utf-8")).hexdigest()
    signa = hmac.new(
        secret_key.encode("utf-8"),
        md5_hash.encode("utf-8"),
        hashlib.sha1,
    ).digest()
    return base64.b64encode(signa).decode("utf-8")


# ---------------------------------------------------------------------------
# 上传
# ---------------------------------------------------------------------------
def upload_file(
    app_id: str,
    secret_key: str,
    file_path: str,
    *,
    language: str = "cn",
    speaker_number: int = 0,
    role_type: int = 1,
    pd: str = "",
) -> dict:
    """
    上传音频文件并返回响应 JSON。

    Args:
        language: 语种, cn=中文(默认), en=英文
        speaker_number: 发音人数, 0=自动判断(默认)
        role_type: 说话人分离, 0=不开启, 1=开启(默认), 2=开启并返回角色名
        pd: 垂直领域个性化, 如 "court"(法院), "edu"(教育), "finance"(金融), "medical"(医疗), "tech"(科技)
    """
    file_path = os.path.abspath(file_path)
    file_size = os.path.getsize(file_path)
    file_name = os.path.basename(file_path)
    ts = int(time.time())
    signa = generate_signa(app_id, secret_key, ts)

    params = {
        "appId": app_id,
        "signa": signa,
        "ts": ts,
        "fileName": file_name,
        "fileSize": file_size,
        "duration": "200",  # 文档要求传一个预估值
        "language": language,
        "audioMode": "fileStream",
        "roleType": role_type,
    }
    if speaker_number > 0:
        params["speakerNumber"] = speaker_number
    if pd:
        params["pd"] = pd

    url = f"{UPLOAD_URL}?{urlencode(params)}"
    logger.info("正在上传文件: %s (%.2f MB)", file_name, file_size / 1024 / 1024)

    with open(file_path, "rb") as f:
        resp = requests.post(
            url,
            data=f,
            headers={"Content-Type": "application/octet-stream"},
            timeout=600,
        )

    result = resp.json()
    if result.get("code") != "000000":
        raise RuntimeError(f"上传失败: {result}")

    order_id = result["content"]["orderId"]
    estimate = result["content"].get("taskEstimateTime", 0)
    logger.info("上传成功, orderId=%s, 预计耗时=%dms", order_id, estimate)
    return result


# ---------------------------------------------------------------------------
# 查询结果
# ---------------------------------------------------------------------------
def get_result(app_id: str, secret_key: str, order_id: str) -> dict:
    """查询单次转写结果。"""
    ts = int(time.time())
    signa = generate_signa(app_id, secret_key, ts)

    params = {
        "appId": app_id,
        "signa": signa,
        "ts": ts,
        "orderId": order_id,
        "resultType": "transfer",
    }
    url = f"{GET_RESULT_URL}?{urlencode(params)}"
    resp = requests.post(url, timeout=30)
    return resp.json()


# ---------------------------------------------------------------------------
# 轮询
# ---------------------------------------------------------------------------
def poll_result(
    app_id: str,
    secret_key: str,
    order_id: str,
    *,
    max_wait: int = 3600,
) -> dict:
    """
    轮询直到转写完成, 返回最终结果。
    采用递增间隔: 5s x6 → 10s x6 → 20s...
    """
    intervals = [5] * 6 + [10] * 6 + [20] * 100
    elapsed = 0
    for interval in intervals:
        result = get_result(app_id, secret_key, order_id)
        if result.get("code") != "000000":
            raise RuntimeError(f"查询失败: {result}")

        status = result["content"]["orderInfo"]["status"]
        if status == STATUS_COMPLETED:
            logger.info("转写完成!")
            return result
        elif status == STATUS_FAILED:
            fail_type = result["content"]["orderInfo"].get("failType", "unknown")
            raise RuntimeError(f"转写失败, orderId={order_id}, failType={fail_type}")
        elif status in (STATUS_CREATED, STATUS_PROCESSING):
            logger.info("转写中... (已等待 %ds)", elapsed)
        else:
            logger.warning("未知状态: %s", status)

        time.sleep(interval)
        elapsed += interval
        if elapsed >= max_wait:
            raise TimeoutError(f"等待超时 ({max_wait}s), orderId={order_id}")

    raise TimeoutError(f"等待超时, orderId={order_id}")


# ---------------------------------------------------------------------------
# 解析结果
# ---------------------------------------------------------------------------
def parse_result(order_result_str: str) -> tuple[str, list[dict]]:
    """
    解析 orderResult JSON 字符串。

    严格按讯飞文档结构：
    - lattice2: 含说话人分离信息的段落列表（优先使用）
    - lattice:  不含说话人信息的段落列表（备选）

    每个 lattice2 条目：
    - begin/end: 绝对时间（单位 10ms）
    - spk: 说话人标签
    - json_1best.st.rt[].ws[].cw[]: 词级数据

    Returns:
        (plain_text, segments)
        - plain_text: 纯文本
        - segments: [{"speaker": "...", "begin": ms, "end": ms, "text": "..."}]
    """
    data = json.loads(order_result_str)
    segments = []

    # 优先使用 lattice2（含说话人信息）
    lattice_key = "lattice2" if "lattice2" in data else "lattice"
    lattice_list = data.get(lattice_key, [])

    for item in lattice_list:
        # lattice2 的时间戳直接在条目上; lattice 的在 st 里
        begin_ms = int(item.get("begin", 0)) * 10  # 10ms → ms
        end_ms = int(item.get("end", 0)) * 10
        speaker = item.get("spk", "")

        json_1best = item.get("json_1best", {})
        if isinstance(json_1best, str):
            json_1best = json.loads(json_1best)

        # 如果 lattice 没有 begin/end，从 st 取
        if begin_ms == 0 and end_ms == 0:
            st = json_1best.get("st", {})
            begin_ms = int(st.get("bg", 0)) * 10
            end_ms = int(st.get("ed", 0)) * 10

        # 拼接词
        st = json_1best.get("st", {})
        words = []
        for rt in st.get("rt", []):
            for ws in rt.get("ws", []):
                for cw in ws.get("cw", []):
                    w = cw.get("w", "")
                    wp = cw.get("wp", "n")
                    if wp != "g":  # g = 分隔符/空白
                        words.append(w)

        text = "".join(words)
        if text.strip():
            segments.append({
                "speaker": speaker,
                "begin": begin_ms,
                "end": end_ms,
                "text": text,
            })

    plain_text = "\n".join(seg["text"] for seg in segments)
    return plain_text, segments


def format_time(ms: int) -> str:
    """毫秒 → HH:MM:SS"""
    total_seconds = ms // 1000
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_transcript(segments: list[dict]) -> str:
    """格式化为带时间戳和说话人的文字稿。

    连续相同说话人的段落会合并为一条，时间取首尾区间。
    """
    if not segments:
        return ""

    # 合并连续相同说话人的段落
    merged = []
    current = {**segments[0]}
    for seg in segments[1:]:
        if seg["speaker"] == current["speaker"]:
            current["end"] = seg["end"]
            current["text"] += seg["text"]
        else:
            merged.append(current)
            current = {**seg}
    merged.append(current)

    lines = []
    for seg in merged:
        time_str = f"[{format_time(seg['begin'])} - {format_time(seg['end'])}]"
        speaker = f"【{seg['speaker']}】" if seg["speaker"] else ""
        lines.append(f"{time_str} {speaker}{seg['text']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def transcribe(
    file_path: str,
    *,
    app_id: str | None = None,
    secret_key: str | None = None,
    language: str = "cn",
    speaker_number: int = 0,
    role_type: int = 1,
    pd: str = "",
    output: str | None = None,
    max_wait: int = 3600,
) -> str:
    """
    录音转文字稿的完整流程。

    Args:
        file_path: 音频文件路径
        app_id: 讯飞 APP_ID（默认从环境变量读取）
        secret_key: 讯飞 SECRET_KEY（默认从环境变量读取）
        language: 语种
        speaker_number: 发音人数
        role_type: 说话人分离
        pd: 垂直领域
        output: 输出文件路径（默认为 音频文件名.txt）
        max_wait: 最大等待秒数

    Returns:
        输出文件的路径
    """
    # 优先加载脚本所在目录的 .env，再回退到当前工作目录
    _script_dir = Path(__file__).parent
    load_dotenv(_script_dir / ".env") or load_dotenv()
    app_id = app_id or os.getenv("XUNFEI_APP_ID")
    secret_key = secret_key or os.getenv("XUNFEI_SECRET_KEY")
    if not app_id or not secret_key:
        raise ValueError("请设置 XUNFEI_APP_ID 和 XUNFEI_SECRET_KEY 环境变量, 或通过参数传入")

    # 1. 上传
    upload_resp = upload_file(
        app_id, secret_key, file_path,
        language=language,
        speaker_number=speaker_number,
        role_type=role_type,
        pd=pd,
    )
    order_id = upload_resp["content"]["orderId"]

    # 2. 轮询
    result = poll_result(app_id, secret_key, order_id, max_wait=max_wait)

    # 3. 解析
    order_result = result["content"]["orderResult"]
    plain_text, segments = parse_result(order_result)

    # 4. 保存
    if output is None:
        stem = Path(file_path).stem
        output = str(Path(file_path).parent / f"{stem}_transcript.txt")

    with open(output, "w", encoding="utf-8") as f:
        f.write(format_transcript(segments))

    # 同时保存纯文本版本
    plain_output = output.replace("_transcript.txt", "_plain.txt")
    with open(plain_output, "w", encoding="utf-8") as f:
        f.write(plain_text)

    # 保存原始 JSON 供调试
    json_output = output.replace("_transcript.txt", "_raw.json")
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info("文字稿已保存: %s", output)
    logger.info("纯文本已保存: %s", plain_output)
    logger.info("原始JSON已保存: %s", json_output)
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="讯飞录音文件转写")
    parser.add_argument("file", help="音频文件路径 (wav/flac/opus/m4a/mp3)")
    parser.add_argument("--output", "-o", help="输出文件路径 (默认: <文件名>_transcript.txt)")
    parser.add_argument("--language", "-l", default="cn", help="语种: cn(中文)/en(英文), 默认 cn")
    parser.add_argument("--speaker-number", "-s", type=int, default=0, help="发音人数, 0=自动判断")
    parser.add_argument("--pd", default="", help="垂直领域: court/edu/finance/medical/tech")
    parser.add_argument("--max-wait", type=int, default=3600, help="最大等待秒数, 默认 3600")
    args = parser.parse_args()

    if not os.path.isfile(args.file):
        print(f"错误: 文件不存在 - {args.file}", file=sys.stderr)
        sys.exit(1)

    try:
        output = transcribe(
            args.file,
            language=args.language,
            speaker_number=args.speaker_number,
            role_type=1,
            pd=args.pd,
            output=args.output,
            max_wait=args.max_wait,
        )
        print(f"\n✅ 转写完成! 文字稿: {output}")
    except Exception as e:
        print(f"\n❌ 转写失败: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
