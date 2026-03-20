# Week 7 LinkedIn Post: Multimodal Models

---

I asked a model to describe a photograph this week. The model had never seen the image — it had never seen *any* image during language model training.

Then I showed it how CLIP works, and suddenly that sentence made complete sense.

**Week 7: Multimodal Models — CLIP, Vision Transformers, and Whisper.**

---

**💡 Aha #1 — CLIP is the ImageNet moment for multimodal AI**

CLIP (Contrastive Language-Image Pre-Training) trains a vision encoder and a text encoder jointly, using contrastive loss on 400 million image-text pairs:

```python
# For a batch of N (image, text) pairs:
image_features = image_encoder(images)   # (N, d)
text_features  = text_encoder(texts)     # (N, d)

# Similarity matrix: NxN
similarity = image_features @ text_features.T / temperature

# Push diagonal (matching pairs) toward 1
# Push off-diagonal (non-matching pairs) toward 0
loss = cross_entropy(similarity, labels=torch.arange(N))
```

After training, CLIP can do zero-shot classification of *any* category — just embed a text description and find the most similar image. It learned "what things look like" from the description of things, not from labelled examples.

---

**💡 Aha #2 — Vision Transformers (ViT) apply self-attention to image patches**

Before ViT, CNNs dominated computer vision. ViT's insight: divide an image into 16×16 pixel patches, flatten each patch into a vector, treat each patch as a "token" — and run a standard Transformer encoder on the sequence.

```
224×224 image → 14×14 = 196 patches → 196 tokens
Each token: a 16×16×3 = 768-dim vector
Run Transformer encoder → patch representations → classification head
```

No convolutions. No inductive biases about local structure. Just attention. And it works better than CNNs at scale.

---

**💡 Aha #3 — Whisper is a supervised sequence-to-sequence model, not a diffusion model**

Audio transcription models like Whisper encode audio spectrograms (frequency × time representations) using a Transformer encoder, then decode to text using a Transformer decoder — exactly like a translation model.

The reason Whisper works so well: it was trained on 680k hours of weakly supervised audio from the internet (audio + subtitle pairs), not clean studio transcriptions. Diversity beats cleanliness at scale.

---

The multimodal theme of this week: alignment between modalities is learned from co-occurrence, not from explicit labels. Images that appear next to text descriptions; audio that appears next to subtitles. The internet, as a co-occurrence machine, is the implicit training signal.

**Do you think we'll see a single model architecture that natively handles text, images, audio, and video with truly unified representations? Or will we always stitch specialised encoders together?**

---

#MultimodalAI #CLIP #VisionTransformer #Whisper #ComputerVision #NLP #MachineLearning #BuildingInPublic #10WeekChallenge
