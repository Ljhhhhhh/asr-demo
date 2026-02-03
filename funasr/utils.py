import re
import asyncio
import aiohttp
import subprocess
from pathlib import Path
from typing import List, Dict, Any

# =============================================================================
# Post-processing Logic (Ported from asr_service/postprocess.py)
# =============================================================================

# 常见语气词列表
FILLER_WORDS = {'呃', '嗯', '啊', '哎', '额', '噢', '哦', '呀', '诶', '唉'}

def clean_filler_words(text: str) -> str:
    """移除语气词（仅移除单独出现或句首的语气词）"""
    for filler in FILLER_WORDS:
        text = re.sub(rf'^{filler}[，、,]?\s*', '', text)
        text = re.sub(rf'[，、,]\s*{filler}[，、,]', '，', text)
    
    # 移除连续的语气词
    for filler in FILLER_WORDS:
        text = re.sub(rf'{filler}{{2,}}', '', text)
    
    return text.strip()

def merge_repetitions(text: str) -> str:
    """合并重复词（如"对对对对"合并为"对"）"""
    text = re.sub(r'([\u4e00-\u9fa5])\1{2,}', r'\1', text)
    text = re.sub(r'([\u4e00-\u9fa5]{1,2})\1{2,}', r'\1', text)
    return text

def apply_itn(text: str) -> str:
    """逆文本标准化：将口语化数字转换为书面形式"""
    cn_nums = {'零': '0', '一': '1', '二': '2', '三': '3', '四': '4',
               '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
               '两': '2', '〇': '0'}
    
    # 百分比转换
    def percent_replace(m):
        num_str = m.group(1)
        result = ''
        for c in num_str:
            if c in cn_nums:
                result += cn_nums[c]
            elif c == '十':
                if not result: result = '1'
                result += '0' if len(result) == 1 else ''
            elif c == '点':
                result += '.'
            else:
                result += c
        return result + '%'
    
    text = re.sub(r'百分之([\u4e00-\u9fa5\d\.]+)', percent_replace, text)
    
    # 年份转换
    def year_replace(m):
        year_cn = m.group(1)
        year_num = ''
        for c in year_cn:
            if c in cn_nums:
                year_num += cn_nums[c]
        return year_num + '年'
    
    text = re.sub(r'([零一二三四五六七八九〇]{4})年', year_replace, text)
    
    # 小数转换
    def decimal_replace(m):
        int_part = m.group(1)
        dec_part = m.group(2)
        
        int_num = 0
        temp = 0
        for c in int_part:
            if c in cn_nums:
                temp = int(cn_nums[c])
            elif c == '十':
                if temp == 0: temp = 1
                int_num += temp * 10
                temp = 0
            elif c == '百':
                int_num += temp * 100
                temp = 0
            elif c == '千':
                int_num += temp * 1000
                temp = 0
        int_num += temp
        
        dec_num = ''
        for c in dec_part:
            if c in cn_nums:
                dec_num += cn_nums[c]
        
        return f'{int_num}.{dec_num}'
    
    text = re.sub(r'([一二三四五六七八九十百千两]+)点([一二三四五六七八九零〇0-9]+)', decimal_replace, text)
    
    # 数量单位转换
    def unit_replace(m):
        num_cn = m.group(1)
        unit = m.group(2)
        num = 0
        temp = 0
        for c in num_cn:
            if c in cn_nums:
                temp = int(cn_nums[c])
            elif c == '十':
                if temp == 0: temp = 1
                num += temp * 10
                temp = 0
            elif c == '百':
                num += temp * 100
                temp = 0
        num += temp
        return f'{num}{unit}'
    
    text = re.sub(r'([一二三四五六七八九十百千两零〇]+)(亿|万|家|个|人)', unit_replace, text)
    
    return text

def postprocess(text: str, apply_itn_flag: bool = True) -> str:
    """完整后处理流程"""
    text = clean_filler_words(text)
    text = merge_repetitions(text)
    if apply_itn_flag:
        text = apply_itn(text)
    text = re.sub(r'[，、]{2,}', '，', text)
    text = re.sub(r'，$', '。', text)
    return text.strip()

# =============================================================================
# Audio Utilities
# =============================================================================

async def download_audio(url: str, save_path: Path):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise Exception(f"Failed to download audio: {response.status}")
            with open(save_path, 'wb') as f:
                while True:
                    chunk = await response.content.read(1024*1024)
                    if not chunk:
                        break
                    f.write(chunk)

async def run_ffmpeg(input_path: Path, output_path: Path, enable_denoise: bool = True) -> None:
    command = [
        "ffmpeg", "-nostdin", "-y",
        "-i", str(input_path),
        "-ac", "1",
        "-ar", "16000",
    ]
    
    if enable_denoise:
        command.extend(["-af", "highpass=f=80,lowpass=f=8000,afftdn=nf=-20"])
    
    command.extend(["-f", "wav", str(output_path)])
    
    await asyncio.to_thread(_run_command, command, "ffmpeg")

def _run_command(command: list[str], name: str) -> None:
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        raise Exception(f"{name} failed: {exc.stderr.decode()}") from exc

