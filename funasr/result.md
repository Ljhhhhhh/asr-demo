# funasr asr result

## 完整响应结构（推荐规范）

当前项目建议 FunASR 服务返回下面结构（result 为核心），其余字段用于兼容与可读性：

```json
{
  "text": "原始文本（可选，保留）",
  "raw_text": "未处理原文（可选，保留）",
  "result": {
    "key": "audio",
    "text": "原始文本（可选，等价 raw_text 或 text）",
    "text_processed": "处理后文本（推荐用作最终全文）",
    "sentence_info": [
      {
        "text": "一句话内容",
        "start": 370,
        "end": 510,
        "spk": 0,
        "timestamp": [
          [370, 400],
          [400, 510]
        ],
        "confidence": 0.88
      }
    ]
  },
  "meta": {
    "time_unit": "ms",
    "model": "paraformer-zh",
    "device": "cuda:0"
  }
}
```

## result 结构详细说明（项目期望）

result: object
• key：string
◦ 固定值 "audio"（可选，但建议保留）
• text：string（可选）
◦ 原始转写文本，可与 raw_text 等价
• text_processed：string（强烈建议）
◦ 经标点/ITN 处理后的最终文本（项目推荐作为 task.text 的来源）
• sentence_info：array（核心字段）
◦ 每条代表一个可用于 SRT 的“段落/句子”
◦ 元素结构如下： sentence_info[i]: object
• text：string（必填）
• start：number（必填，单位毫秒）
• end：number（必填，单位毫秒）
• spk：number|string（可选，说话人编号）
• timestamp：array<[number, number]>（可选，字级/词级时间戳）
• confidence：number（可选）

## 典型案例（精简示例）

```json
{
  "text": "今天一一个降文。嗯，对啊对对，是的。",
  "raw_text": "今天一一个降文。嗯，对啊对对，是的。",
  "result": {
    "key": "audio",
    "text": "今天一一个降文。嗯，对啊对对，是的。",
    "text_processed": "今天1个降文。嗯，对啊对对，是的。",
    "sentence_info": [
      {
        "text": "今天一一个降文。",
        "start": 370,
        "end": 1090,
        "spk": 0,
        "timestamp": [
          [370, 510],
          [510, 750],
          [950, 1090]
        ],
        "confidence": 0.91
      },
      {
        "text": "嗯，对啊对对，是的。",
        "start": 1090,
        "end": 1650,
        "spk": 2,
        "timestamp": [
          [1090, 1270],
          [1270, 1410],
          [1410, 1650]
        ],
        "confidence": 0.87
      }
    ]
  },
  "meta": {
    "time_unit": "ms",
    "model": "paraformer-zh",
    "device": "cuda:0"
  }
}
```

## 当前代码的对齐点（提醒）

• 核心分段应从 result.sentence_info 解析
这是你 asr_result.json 的真实结构。
• 时间单位
你的 start/end 当前是毫秒，推荐 meta.time_unit = "ms"。
• 最终全文
推荐使用 result.text_processed 作为 ai_asr_task.text。
如果你确认这就是 FunASR 服务的“标准返回”，我可以直接把解析逻辑调整到该结构，并同时保留对 text/segments/meta 的兼容。
