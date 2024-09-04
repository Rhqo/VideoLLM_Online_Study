# VideoLLM-online: Online Video Large Language Model for Streaming Video

# 1. Introduction

다양한 상황에서 인간에게 적극적으로 도움을 줄 수 있는, 입력을 episodic memory로 digitize하고, 온라인의 연속적인 환경에서 predict를 할 수 있는, always-on, contextual AI assistant를 구축하는 것은 AI연구에서 중요한 임무 중 하나이다.

LLM의 발전에 따라 LMM또한 발전되어 왔고, vision-language dialogue, spatial understanding, processing diver modalities 등의 분야에서의 능력이 입증된 바 있다.

OpenAI의 GPT-4V나 GPT-4o는 특히 다재다능한 AI assistant가 되어가고 있다.

하지만, 가장 발전된 GPT-4o조차도 streaming voice기반의 multimodal assitance에 그쳤다.

이제는 video stream 내에서 자유 형식의 user-assistant 대화를 지원하는, always on, contextual인 J.A.R.V.I.S와 같은 비디오 비서를 구상해야 할 때이며, 이를 "Video Streaming Dialogue"라는 용어로 부르겠다.

기존의 offline이며, short-video clip에서만 작동하는 video understanding을 위한 LMM들과 달리, online assistant는 계속해서 refresh되는 연속적인 video frame을 받아야 한다.

이는 새로운 과제를 제시한다.

1. User query가 **temporally aligned**되어야 하며, 따라서 VideoLLM은 event를 놓치지 않기 위해 video-level의 응답을 하는 것이 아닌, 매 frame을 scan해야 한다.
2. Summarization, planning의 질문에 대해 응답하기 위해 VideoLLLM은 **long-context**한 과거의 시각과 언어를 유지해야 한다. 이는 LLM context window의 최대 창을 넘어버릴 수 있고, causal decoding 속도와 GPU 메모리에 부담이 된다.
3. VideoLLM은 응답을 **real-time**으로 생성해야 하며, 시나리오 안에서 항상 켜져(always-on) 있어야 한다.

기존의 Vision language 모델에서 영감을 얻은 온라인 VideoLLM 개발의 한 가지 가능성은 video stream 내에서 프레임별 채팅을 달성하기 위해 multi-turn dialogue 형식을 사용하는 것이다.

이는 각 타임스탬프에서 visual frame을 query로 활용하여 매우 빈번한 사용자 상호작용을 유도함으로써 달성할 수 있다.

우리는 이를 GPT-4V에 대한 프롬프트 엔지니어링에 사용했지만, 결과가 좋지 않았다.

GPT-4V는 각 프레임마다 길이가 긴 내용을 출력하는 경향이 있어 상당한 지연이 발생하며, 실시간 스트리밍 비디오에는 맞지 않았다.

우리는 또한 프레임별 채팅을 위한 baseline model을 훈련하는 것도 시도했지만, 이 접근 방식은 많은 중복된 프레임에서 유해한 언어 모델링이 발생하여 언어 모델링 기능이 현저히 저하되었다.

이에 따라 우리는 Learning-In-Video-strEam (LIVE) 라는, online video assistant를 위한 학습, 데이터, 추론 method들이 포함된 종합 Framework를 제안한다.

Per-frame dialogue approach와 달리 LIVE는 Streaming EOS라는, 모델이 video stream에서 언제 대답하고 언제 조용해야 하는지를 학습할 수 있게 하는 training object를 사용한다.

EOS token은 input/output sequence에 나타나지 않기 때문에, next-token prediction과 다르다.

하지만 학습을 위한 autoregressive loss를 사용할 수 있다.

이러한 디자인은 불필요한 context를 줄이고, 긴 streaming video를 관리하는 것을 쉽게 한다.

그럼에도 불구하고, 학습에는 video stream에서의 user query들과 assistant response들이 필요하다.

이를 다루기 위해서, LIVE는 offline annotation으로부터 online dialogue들로 변환하는 streaming dialogue generation scheme을 사용한다.

Inference 효율을 높이기 위해, key-value caching을 사용하고, bottleneck을 방지하기 위한 fast visual encoding, slow language decoding을 사용한다.

LIVE framework를 통해, CLIP vision encoder와 Llama2/Llama-3 language model을 사용하여 간단한 VideoLLM online model을 구현한다.

Video streaming dialogue를 evaluate하기 위해

# 2. Related Work

### Visual Dialogue

### Large Multimodal Models

### Online Video Understanding

### Efficient Token Decoding

# 3. Method

## 3.1 Video Streaming Dialogue

### Problem Formulation

### Interleaved/Per-frame Dialogue are Suboptimal

### Streaming EOS Prediction

## 3.2 Data

### Online Annotations to Video Streaming Dialogue

### Offline Annotations to Video Streaming Dialogue

## 3.3 Model Training

### Model Architecture

### Training Loss

## 3.4 Inference

### Probability Correction

### Continuous Key-Value Cache

### Parallelization of encoding and decoding

# 4. Experiments

## 4.1 Implementation Details

## 4.2 Evaluation Setting

### Datasets

### Evaluation metrics

### Baselines

## 4.3 Ablation Study

### Learning Method

### Streaming Loss

### Inference Efficiency

## 4.4 Results

### Offline Language Modeling

### Comparison between Model Variants

### Visualization

# 5. Conclusion
