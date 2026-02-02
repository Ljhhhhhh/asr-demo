const form = document.getElementById('upload-form');
const fileInput = document.getElementById('file-input');
const hotwordsInput = document.getElementById('hotwords-input');
const postprocessCheckbox = document.getElementById('postprocess-checkbox');
const statusEl = document.getElementById('status');
const transcriptEl = document.getElementById('transcript');
const copyButton = document.getElementById('copy-button');
const uploadButton = document.getElementById('upload-button');

// Speaker colors for visual distinction
const SPEAKER_COLORS = [
  '#3b82f6',
  '#10b981',
  '#f59e0b',
  '#ef4444',
  '#8b5cf6',
  '#ec4899',
  '#06b6d4',
  '#84cc16',
];

const setStatus = (message, tone = 'info') => {
  statusEl.textContent = message;
  statusEl.dataset.tone = tone;
};

const formatTime = (ms) => {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
};

const renderTranscript = (result) => {
  transcriptEl.innerHTML = '';

  // Check if we have sentence_info with timestamps and speaker info
  const sentences = result.sentence_info || [];

  if (sentences.length > 0) {
    sentences.forEach((sentence) => {
      const item = document.createElement('div');
      item.className = 'transcript-item';

      // Time badge
      const timeBadge = document.createElement('span');
      timeBadge.className = 'time-badge';
      const startTime = formatTime(sentence.start || 0);
      const endTime = formatTime(sentence.end || 0);
      timeBadge.textContent = `${startTime} - ${endTime}`;

      // Speaker badge (if available)
      const speakerId = sentence.spk !== undefined ? sentence.spk : null;
      if (speakerId !== null) {
        const speakerBadge = document.createElement('span');
        speakerBadge.className = 'speaker-badge';
        speakerBadge.textContent = `Speaker ${speakerId + 1}`;
        speakerBadge.style.backgroundColor =
          SPEAKER_COLORS[speakerId % SPEAKER_COLORS.length];
        item.appendChild(speakerBadge);
      }

      item.appendChild(timeBadge);

      // Text content
      const textSpan = document.createElement('span');
      textSpan.className = 'transcript-text';
      textSpan.textContent = sentence.text || '';
      item.appendChild(textSpan);

      transcriptEl.appendChild(item);
    });
  } else {
    // Fallback to plain text
    const textDiv = document.createElement('div');
    textDiv.className = 'transcript-plain';
    textDiv.textContent = result.text || '';
    transcriptEl.appendChild(textDiv);
  }

  copyButton.disabled = !result.text;
};

const getPlainText = () => {
  const items = transcriptEl.querySelectorAll('.transcript-item');
  if (items.length > 0) {
    return Array.from(items)
      .map((item) => {
        const timeBadge = item.querySelector('.time-badge');
        const speakerBadge = item.querySelector('.speaker-badge');
        const textSpan = item.querySelector('.transcript-text');

        const time = timeBadge ? timeBadge.textContent : '';
        const speaker = speakerBadge ? speakerBadge.textContent : '';
        const text = textSpan ? textSpan.textContent : '';

        // Format: [00:00 - 00:30] Speaker 1: 文本内容
        if (speaker) {
          return `[${time}] ${speaker}: ${text}`;
        }
        return `[${time}] ${text}`;
      })
      .join('\n\n');
  }
  const plain = transcriptEl.querySelector('.transcript-plain');
  return plain ? plain.textContent : '';
};

copyButton.addEventListener('click', () => {
  const text = getPlainText().trim();
  if (!text) {
    return;
  }
  navigator.clipboard.writeText(text).then(() => {
    setStatus('Transcript copied.');
  });
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file) {
    setStatus('Please select an audio file.', 'error');
    return;
  }

  setStatus('Uploading and transcribing...');
  transcriptEl.innerHTML = '';
  uploadButton.disabled = true;
  copyButton.disabled = true;

  const body = new FormData();
  body.append('file', file);
  body.append('hotwords', hotwordsInput.value || '');
  body.append('enable_postprocess', postprocessCheckbox.checked);

  try {
    const response = await fetch('/asr/transcribe', {
      method: 'POST',
      body,
    });

    const payload = await response.json();
    if (!response.ok) {
      const errorMessage =
        payload.detail || payload.error || 'Transcription failed.';
      setStatus(errorMessage, 'error');
      return;
    }

    renderTranscript(payload.result || { text: payload.text });
    setStatus('Done.');
  } catch (error) {
    setStatus('Network error. Please try again.', 'error');
  } finally {
    uploadButton.disabled = false;
  }
});
