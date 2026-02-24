# 讯飞录音文件转写

基于[讯飞开放平台 - 录音文件转写（标准版）](https://www.xfyun.cn/doc/asr/ifasr_new/API.html) API 实现。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置密钥

复制 `.env.example` 为 `.env`，填入讯飞平台的凭证：

```bash
cp .env.example .env
# 编辑 .env，填入 APP_ID 和 SECRET_KEY
```

### 3. 运行转写

```bash
# 基本用法
python transcribe.py 音频文件.m4a

# 指定输出路径
python transcribe.py 音频文件.m4a -o 输出.txt

# 指定发音人数（提高说话人分离准确度）
python transcribe.py 音频文件.m4a -s 2

# 英文音频
python transcribe.py audio.mp3 -l en
```

### 输出文件

转写完成后会生成三个文件：

| 文件               | 说明                     |
| ------------------ | ------------------------ |
| `*_transcript.txt` | 带时间戳和说话人的文字稿 |
| `*_plain.txt`      | 纯文本（无时间戳）       |
| `*_raw.json`       | 讯飞原始返回 JSON        |

### CLI 参数

| 参数                   | 说明             | 默认值                    |
| ---------------------- | ---------------- | ------------------------- |
| `file`                 | 音频文件路径     | 必填                      |
| `-o, --output`         | 输出文件路径     | `<文件名>_transcript.txt` |
| `-l, --language`       | 语种 (cn/en)     | cn                        |
| `-s, --speaker-number` | 发音人数, 0=自动 | 0                         |
| `--pd`                 | 垂直领域         | 空                        |
| `--max-wait`           | 最大等待秒数     | 3600                      |

## 支持的音频格式

wav, flac, opus, m4a, mp3（5 小时以内）
