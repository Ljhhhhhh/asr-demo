"""ASR 后处理模块

提供文字稿质量优化功能：
- 语气词过滤
- 重复词合并
- 段落合并（同一说话人相邻发言）
"""

import re
from typing import List, Dict, Any


# 常见语气词列表
FILLER_WORDS = {'呃', '嗯', '啊', '哎', '额', '噢', '哦', '呀', '诶', '唉'}


def clean_filler_words(text: str) -> str:
    """
    移除语气词（仅移除单独出现或句首的语气词）
    
    Args:
        text: 原始文本
        
    Returns:
        清理后的文本
    """
    # 移除句首语气词 + 标点
    for filler in FILLER_WORDS:
        text = re.sub(rf'^{filler}[，、,]?\s*', '', text)
        text = re.sub(rf'[，、,]\s*{filler}[，、,]', '，', text)
    
    # 移除连续的语气词
    for filler in FILLER_WORDS:
        text = re.sub(rf'{filler}{{2,}}', '', text)
    
    return text.strip()


def merge_repetitions(text: str) -> str:
    """
    合并重复词（如"对对对对"合并为"对"）
    
    Args:
        text: 原始文本
        
    Returns:
        合并后的文本
    """
    # 匹配连续重复的单个汉字（3次以上）
    text = re.sub(r'([\u4e00-\u9fa5])\1{2,}', r'\1', text)
    
    # 匹配连续重复的短词组（如"是是"、"对对"）—— 保留2次
    text = re.sub(r'([\u4e00-\u9fa5]{1,2})\1{2,}', r'\1', text)
    
    return text


def merge_speaker_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    合并同一说话人的相邻片段
    
    Args:
        segments: 带时间戳和说话人的片段列表
            每个片段格式: {"start": float, "end": float, "speaker": str, "text": str}
            
    Returns:
        合并后的片段列表
    """
    if not segments:
        return []
    
    merged = []
    current = segments[0].copy()
    
    for seg in segments[1:]:
        # 如果是同一说话人且时间间隔小于2秒，合并
        if (seg.get('speaker') == current.get('speaker') and 
            seg.get('start', 0) - current.get('end', 0) < 2.0):
            current['end'] = seg.get('end', current['end'])
            current['text'] = current.get('text', '') + seg.get('text', '')
        else:
            merged.append(current)
            current = seg.copy()
    
    merged.append(current)
    return merged


def apply_itn(text: str) -> str:
    """
    逆文本标准化：将口语化数字转换为书面形式
    
    基于规则的简单实现，覆盖常见场景。
    
    Args:
        text: 原始文本
        
    Returns:
        标准化后的文本
    """
    # 中文数字映射
    cn_nums = {'零': '0', '一': '1', '二': '2', '三': '3', '四': '4',
               '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
               '两': '2', '〇': '0'}
    
    # 百分比转换：百分之X -> X%
    def percent_replace(m):
        num_str = m.group(1)
        # 简单转换
        result = ''
        for c in num_str:
            if c in cn_nums:
                result += cn_nums[c]
            elif c == '十':
                if not result:
                    result = '1'
                result += '0' if len(result) == 1 else ''
            elif c == '点':
                result += '.'
            else:
                result += c
        return result + '%'
    
    text = re.sub(r'百分之([\u4e00-\u9fa5\d\.]+)', percent_replace, text)
    
    # 年份转换：二零二五年 -> 2025年
    def year_replace(m):
        year_cn = m.group(1)
        year_num = ''
        for c in year_cn:
            if c in cn_nums:
                year_num += cn_nums[c]
        return year_num + '年'
    
    text = re.sub(r'([零一二三四五六七八九〇]{4})年', year_replace, text)
    
    # 简单数字转换：X点X（如 二十五点六）
    def decimal_replace(m):
        int_part = m.group(1)
        dec_part = m.group(2)
        
        # 转换整数部分
        int_num = 0
        temp = 0
        for c in int_part:
            if c in cn_nums:
                temp = int(cn_nums[c])
            elif c == '十':
                if temp == 0:
                    temp = 1
                int_num += temp * 10
                temp = 0
            elif c == '百':
                int_num += temp * 100
                temp = 0
            elif c == '千':
                int_num += temp * 1000
                temp = 0
        int_num += temp
        
        # 转换小数部分
        dec_num = ''
        for c in dec_part:
            if c in cn_nums:
                dec_num += cn_nums[c]
        
        return f'{int_num}.{dec_num}'
    
    text = re.sub(r'([一二三四五六七八九十百千两]+)点([一二三四五六七八九零〇\\d]+)', decimal_replace, text)
    
    # X亿/万 转换
    def unit_replace(m):
        num_cn = m.group(1)
        unit = m.group(2)
        
        # 简单转换
        num = 0
        temp = 0
        for c in num_cn:
            if c in cn_nums:
                temp = int(cn_nums[c])
            elif c == '十':
                if temp == 0:
                    temp = 1
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
    """
    完整后处理流程
    
    Args:
        text: 原始 ASR 输出文本
        apply_itn_flag: 是否应用 ITN 标准化
        
    Returns:
        处理后的文本
    """
    # 1. 清理语气词
    text = clean_filler_words(text)
    
    # 2. 合并重复词
    text = merge_repetitions(text)
    
    # 3. ITN 标准化
    if apply_itn_flag:
        text = apply_itn(text)
    
    # 4. 清理多余标点
    text = re.sub(r'[，、]{2,}', '，', text)
    text = re.sub(r'，$', '。', text)
    
    return text.strip()
