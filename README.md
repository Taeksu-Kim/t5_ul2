# T5, UL2

## Paper

T5(Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer)   
https://arxiv.org/pdf/1910.10683.pdf

UL2(Unified Language Model Pre-training for Natural Language Understanding and Generation)   
https://arxiv.org/pdf/1905.03197.pdf

## Processor
- huggingface의 T5 pre-train 코드 중 데이터 레이블링 코드는 Max_length에 따라 mask가 부여되므로 인풋 길이에 따라 다르게 부여되도록 수정하여 구현
- Mixture-of-Denoiser(MoD)구현(UL2)

## Train
- Huggingface의 TPU 기반 Flax pre-train 코드를 수정하여 구현

## Reference
https://huggingface.co/transformers/v4.1.1/model_doc/t5.html   
https://github.com/paust-team/pko-t5   
https://wandb.ai/yepster/tpu-t5-base/reports/Adafactor-learning-rate-0-005-seems-best-for-t5-base-training--VmlldzoxNTgyODIw   
https://junbuml.ee/KcT5-Pretraining-on-TPU-with-flax   
https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/README.md#t5-like-span-masked-language-modeling   
 
