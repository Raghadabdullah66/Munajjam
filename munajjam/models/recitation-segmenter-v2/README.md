---
base_model: facebook/w2v-bert-2.0
library_name: transformers
license: mit
metrics:
- accuracy
- f1
- precision
- recall
tags:
- generated_from_trainer
- arabic
- quran
- speech-segmentation
model-index:
- name: recitation-segmenter-v2
  results: []
pipeline_tag: automatic-speech-recognition
language: ar
---

# recitation-segmenter-v2: Quranic Recitation Segmenter

This model is a fine-tuned version of [facebook/w2v-bert-2.0](https://huggingface.co/facebook/w2v-bert-2.0) for segmenting Holy Quran recitations based on pause points (waqf). It was presented in the paper [Automatic Pronunciation Error Detection and Correction of the Holy Quran's Learners Using Deep Learning](https://huggingface.co/papers/2509.00094).

Project Page: https://obadx.github.io/prepare-quran-dataset/
GitHub Repository: https://github.com/obadx/recitations-segmenter

It achieves the following results on the evaluation set:
- Accuracy: 0.9958
- F1: 0.9964
- Loss: 0.0132
- Precision: 0.9976
- Recall: 0.9951

## Model description

The `recitation-segmenter-v2` model is an enhanced AI model capable of segmenting Holy Quran recitations based on pause points (`waqf`) with high accuracy. It is built upon a fine-tuned [Wav2Vec2Bert](https://huggingface.co/docs/transformers/model_doc/wav2vec2-bert) model, performing Sequence Frame Level Classification with a 20-millisecond resolution. This model and its accompanying Python library are designed for high-performance processing of any number and length of Quranic recitations, from a few seconds to several hours, without performance degradation.

Key Features:
*   Segments Quranic recitations according to `waqf` (pause rules).
*   Specifically trained for Quranic recitations.
*   High accuracy, up to 20 milliseconds precision.
*   Requires only ~3 GB of GPU memory.
*   Capable of processing recitations of any duration without performance loss.

The model is part of a larger effort described in the associated paper, aiming to bridge gaps in assessing spoken language for the Holy Quran. This includes an automated pipeline to produce high-quality Quranic datasets and a novel ASR-based approach for pronunciation error detection using a custom Quran Phonetic Script (QPS).

## Intended uses & limitations

This model is primarily intended for:
*   Automatic segmentation of Holy Quran recitations for educational purposes or content analysis.
*   Building high-quality Quranic audio databases.
*   As a foundational component for larger systems focused on pronunciation error detection and correction for Quran learners.

**Limitations**:
*   The segmenter currently considers `sakt` (a very short pause without breath) as a full `waqf` (stop), which might be a nuance for advanced Tajweed analysis.
*   The model is specifically trained and optimized for Quranic recitations and might not generalize well to other forms of spoken Arabic.

## Training and evaluation data

The model was fine-tuned on a meticulously collected dataset of Quranic recitations. The data collection process, described in the associated paper, involved a 98% automated pipeline including collection from expert reciters, segmentation at pause points (`waqf`) using a fine-tuned `wav2vec2-BERT` model, transcription of segments, and transcript verification via a novel Tasmeea algorithm. The dataset comprises over 850 hours of audio (~300K annotated utterances).

The data preparation involved:
1.  Downloading Quranic recitations and converting them to Hugging Face Audio Dataset format at 16000 Hz sample rate.
2.  Pre-segmenting verses based on pauses using `sliero-vad-v4` from [everyayah.com](https://everyayah.com).
3.  Applying post-processing (e.g., `min_silence_duration_ms`, `min_speech_duration_ms`, `pad_duration_ms`) to refine segments and manual verification for high-quality divisions.
4.  Applying data augmentation techniques, including time stretching (speeding up/slowing down 40% of recitations) and various audio effects (Aliasing, AddGaussianNoise, BandPassFilter, PitchShift, RoomSimulator, etc.) using the `audiomentations` library.
5.  Normalizing audio segments to 16000 Hz and chunking them, with a maximum length of 20 seconds, using a sliding window approach for longer segments.

The training dataset and its augmented version are available on Hugging Face:
*   [Training Data](https://huggingface.co/datasets/obadx/recitation-segmentation)
*   [Augmented Training Data](https://huggingface.co/datasets/obadx/recitation-segmentation-augmented)

## Usage

You can use this model with its accompanying Python library, `recitations-segmenter`, which integrates with Hugging Face `transformers`.

First, ensure `ffmpeg` and `libsoundfile` are installed system-wide.

### Requirements

Install `ffmpeg` and `libsoundfile` system-wide.

#### Linux

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libsndfile1 portaudio19-dev
```

#### Windows & Mac

You can create an `anaconda` environment and then install these libraries:

```bash
conda create -n segment python=3.12
conda activate segment
conda install -c conda-forge ffmpeg libsndfile
```

### Via pip

```bash
pip install recitations-segmenter
```

### Sample usage (Python API)

Here's a complete example for using the library in Python. A Google Colab example is also available: [Open in Colab](https://colab.research.google.com/drive/1-RuRQOj4l2MA_SG2p4m-afR7MAsT5I22?usp=sharing)

```python
from pathlib import Path

from recitations_segmenter import segment_recitations, read_audio, clean_speech_intervals
from transformers import AutoFeatureExtractor, AutoModelForAudioFrameClassification
import torch

if __name__ == '__main__':
    device = torch.device('cuda')
    dtype = torch.bfloat16

    processor = AutoFeatureExtractor.from_pretrained(
        "obadx/recitation-segmenter-v2")
    model = AutoModelForAudioFrameClassification.from_pretrained(
        "obadx/recitation-segmenter-v2",
    )

    model.to(device, dtype=dtype)

    # Change this to the file pathes of Holy Quran recitations
    # File pathes with the Holy Quran Recitations
    file_pathes = [
        './assets/dussary_002282.mp3',
        './assets/hussary_053001.mp3',
    ]
    waves = [read_audio(p) for p in file_pathes]

    # Extracting speech inervals in samples according to 16000 Sample rate
    sampled_outputs = segment_recitations(
        waves,
        model,
        processor,
        device=device,
        dtype=dtype,
        batch_size=8,
    )

    for out, path in zip(sampled_outputs, file_pathes):
        # Clean The speech intervals by:
        # * merging small silence durations
        # * remove small speech durations
        # * add padding to each speech duration
        # Raises:
        # * NoSpeechIntervals: if the wav is complete silence
        # * TooHighMinSpeechDruation: if `min_speech_duration` is too high which
        # resuls for deleting all speech intervals
        clean_out = clean_speech_intervals(
            out.speech_intervals,
            out.is_complete,
            min_silence_duration_ms=30,
            min_speech_duration_ms=30,
            pad_duration_ms=30,
            return_seconds=True,
        )

        print(f'Speech Intervals of: {Path(path).name}: ')
        print(clean_out.clean_speech_intervals)
        print(f'Is Recitation Complete: {clean_out.is_complete}')
        print('-' * 40)
```

## Training procedure

The model was trained on `Wav2Vec2BertForAudioFrameClassification` using the `transformers` library. More detailed motivations, methodology, and setup can be found in the GitHub repository's "تفاصيل التدريب" section.

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 50
- eval_batch_size: 64
- seed: 42
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: constant
- lr_scheduler_warmup_ratio: 0.2
- num_epochs: 1

### Training results

| Training Loss | Epoch  | Step | Accuracy | F1     | Validation Loss | Precision | Recall |
|:-------------:|:------:|:----:|:--------:|:------:|:---------------:|:---------:|:------:|
| 0.0701        | 0.2507 | 275  | 0.9953   | 0.9959 | 0.0249          | 0.9947    | 0.9971 |
| 0.0234        | 0.5014 | 550  | 0.9953   | 0.9959 | 0.0185          | 0.9940    | 0.9977 |
| 0.0186        | 0.7521 | 825  | 0.9958   | 0.9964 | 0.0132          | 0.9976    | 0.9951 |

### Framework versions

- Transformers 4.51.3
- Pytorch 2.2.1+cu121
- Datasets 3.5.0
- Tokenizers 0.21.1

## Citation
If you find our work helpful or inspiring, please feel free to cite it.

```bibtex
@article{ibrahim2025automatic,
  title={Automatic Pronunciation Error Detection and Correction of the Holy Quran's Learners Using Deep Learning},
  author={Abdullah Abdelfattah, Mahmoud I.Khalil, Hazem M.Abbas},
  journal={arXiv preprint arXiv:2509.00094},
  year={2025}
}
```