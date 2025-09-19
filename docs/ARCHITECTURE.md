# Architecture Diagram - Subtitle Translator

## Component Architecture

```
┌─────────────────────────────────────────┐
│            User Interface               │
│  ┌─────────────┐  ┌──────────────────┐  │
│  │ Command Line│  │ Rich Progress UI │  │
│  └─────────────┘  └──────────────────┘  │
└─────────────────────────────────────────┘
               │
┌─────────────────────────────────────────┐
│           Processing Cores              │
│  ┌─────────────┐  ┌──────────────────┐  │
│  │whisper_clean│  │asr_translate_srt │  │
│  │    .py      │  │      .py         │  │
│  └─────────────┘  └──────────────────┘  │
│         │                 │             │
│  ┌─────────────┐          │             │
│  │whisper_mvp  │          │             │
│  │    .py      │          │             │
│  └─────────────┘          │             │
└─────────────────────────────────────────┘
      │                │
┌─────────────┐  ┌──────────────────┐
│ Media Input │  │   ML Models      │
│  Processing │  │                  │
│  ┌───────┐  │  │  ┌────────────┐  │
│  │ FFmpeg│  │  │  │ Whisper ASR│  │
│  └───────┘  │  │  └────────────┘  │
│             │  │  ┌────────────┐  │
│             │  │  │ NLLB/M2M100│  │
│             │  │  │ Translation│  │
│             │  │  └────────────┘  │
└─────────────┘  └──────────────────┘
```

## Processing Pipeline

```
┌─────────┐     ┌───────────────┐     ┌────────────────┐
│  Input  │     │ Audio         │     │ Whisper Model  │
│  Video  │────>│ Extraction    │────>│ Transcription  │
│  File   │     │ (FFmpeg)      │     │                │
└─────────┘     └───────────────┘     └────────────────┘
                                             │
                                             │
                 ┌───────────────┐     ┌────────────────┐
                 │ SRT File      │<────│ Optional       │
                 │ Generation    │     │ Translation    │
                 └───────────────┘     └────────────────┘
```

## Two-Pass Translation Pipeline (asr_translate_srt.py)

```
┌─────────┐    ┌───────────────┐    ┌────────────────┐
│  Input  │    │ Audio         │    │ ASR            │
│  Video  │───>│ Extraction    │───>│ Transcription  │
│  File   │    │ (FFmpeg)      │    │                │
└─────────┘    └───────────────┘    └────────────────┘
                                           │
                                           ▼
                                    ┌────────────────┐    ┌────────────────┐
                                    │ External MT    │    │ Fallback to    │
                                    │ Translation    │───>│ Whisper        │
                                    │ (if available) │    │ Translation    │
                                    └────────────────┘    └────────────────┘
                                           │                     │
                                           └─────────┬───────────┘
                                                     ▼
                                            ┌─────────────────────┐
                                            │ Subtitle Quality    │
                                            │ Shaping:            │
                                            │ - Line wrapping     │
                                            │ - Max CPS           │
                                            │ - Min duration      │
                                            │ - Gap handling      │
                                            └─────────────────────┘
                                                     │
                                                     ▼
                                            ┌─────────────────────┐
                                            │ SRT File            │
                                            │ Generation          │
                                            └─────────────────────┘
```

## Dependency Structure

```
┌──────────────────────┐
│  Subtitle Translator │
└──────────────────────┘
           │
   ┌───────┴────────┐
   ▼                ▼
┌─────────┐  ┌────────────┐
│ Python  │  │ External   │
│ Deps    │  │ Binaries   │
└─────────┘  └────────────┘
   │              │
   │         ┌────┴───────┐
   │         ▼            ▼
   │    ┌────────┐  ┌───────────┐
   │    │ FFmpeg │  │CUDA+cuDNN │
   │    └────────┘  └───────────┘
   │
┌──┴───────────────┐
▼                  ▼
┌──────────────┐  ┌────────────┐
│faster-whisper│  │ rich       │
└──────────────┘  └────────────┘
   │
   ├─────────────┐
   ▼             ▼
┌─────────┐  ┌─────────┐
│ctranslate2  │tokenizers│
└─────────┘  └─────────┘
```

## GPU Fallback Logic

```
┌─────────────┐
│ Load Model  │
│ with GPU    │
└─────────────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐
│ Try GPU     │ No  │ Check       │
│ Execution   │────>│ --no-fallback│
└─────────────┘     └─────────────┘
       │                  │
  Error│              Yes │     No
       ▼                  ▼      ▼
┌─────────────┐     ┌──────────┐ │
│ Load Model  │     │ Raise    │ │
│ with CPU    │     │ Exception│ │
└─────────────┘     └──────────┘ │
       │                         │
       │                         │
       ▼                         ▼
┌─────────────┐           ┌──────────┐
│ Complete    │           │ Exit with│
│ with CPU    │           │ Error    │
└─────────────┘           └──────────┘
```

## File Processing Flow

```
┌─────────────┐
│ Input Path  │
└─────────────┘
       │
       ▼
┌─────────────────────┐
│ Is it a directory?  │
└─────────────────────┘
       │
┌──────┴───────┐
│              │
▼              ▼
┌────────┐  ┌──────────────────┐
│ Single │  │ Recursive Search │
│ File   │  │ for Media Files  │
└────────┘  └──────────────────┘
    │              │
    └──────────────┘
           │
           ▼
┌─────────────────────────┐
│ For Each File:          │
│ 1. Extract Audio        │
│ 2. Transcribe/Translate │
│ 3. Generate SRT         │
│ 4. (Optional) Remux     │
└─────────────────────────┘
```