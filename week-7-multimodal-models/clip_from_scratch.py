"""
clip_from_scratch.py
====================
Implements CLIP (Contrastive Language-Image Pre-Training) architecture
and demonstrates zero-shot image classification.

CLIP trains two encoders jointly:
- Image encoder: ViT or ResNet → d-dim embedding
- Text encoder: Transformer → d-dim embedding

Training: contrastive loss on 400M (image, text) pairs from the internet.
After training: any image and any text can be compared in the same space.

Zero-shot classification:
1. Embed all class name text descriptions
2. Embed the test image
3. Find the text embedding with highest cosine similarity
4. That's the predicted class — no fine-tuning needed

Usage:
    python clip_from_scratch.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Vision Transformer Encoder (simplified)
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """
    Split an image into patches and project each patch to embedding_dim.
    
    A 224×224 image with 16×16 patches:
    - (224 / 16)² = 196 patches
    - Each patch: 16 × 16 × 3 = 768 pixels → linear projection → embedding_dim
    """
    def __init__(
        self,
        image_size:    int = 224,
        patch_size:    int = 16,
        in_channels:   int = 3,
        embedding_dim: int = 512,
    ):
        super().__init__()
        self.patch_size  = patch_size
        self.n_patches   = (image_size // patch_size) ** 2
        self.projection  = nn.Conv2d(in_channels, embedding_dim, patch_size, stride=patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: image (B, C, H, W)
        Returns:
            patches: (B, n_patches, embedding_dim)
        """
        # Conv2d with kernel=patch_size, stride=patch_size extracts patches efficiently
        x = self.projection(x)                    # (B, D, H/P, W/P)
        x = x.flatten(2)                          # (B, D, n_patches)
        x = x.transpose(1, 2)                     # (B, n_patches, D)
        return x


class ViTEncoder(nn.Module):
    """
    Vision Transformer encoder for CLIP.
    
    Key additions over standard Transformer:
    - [CLS] token: a learnable token prepended to the patch sequence.
      After all attention layers, [CLS] aggregates global image information.
    - Position embeddings: since patches have no inherent order (unlike text),
      we add learnable position embeddings to each patch token.
    """
    def __init__(
        self,
        image_size:     int = 64,   # Smaller for demo
        patch_size:     int = 8,
        in_channels:    int = 3,
        embedding_dim:  int = 256,
        n_heads:        int = 4,
        n_layers:       int = 4,
        mlp_ratio:      float = 4.0,
        dropout:        float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embedding_dim)
        n_patches = (image_size // patch_size) ** 2
        
        # [CLS] token: learnable global representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim) * 0.02)
        # Learnable position embeddings for each patch + [CLS]
        self.pos_emb   = nn.Parameter(torch.randn(1, n_patches + 1, embedding_dim) * 0.02)
        self.dropout   = nn.Dropout(dropout)
        
        # Standard Transformer encoder
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=n_heads,
            dim_feedforward=int(embedding_dim * mlp_ratio),
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm        = nn.LayerNorm(embedding_dim)
        self.proj        = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: images (B, C, H, W)
        Returns:
            image embeddings (B, embedding_dim), L2 normalised
        """
        B = x.shape[0]
        patches = self.patch_embed(x)              # (B, n_patches, D)
        
        # Prepend [CLS] token to each image in the batch
        cls = self.cls_token.expand(B, -1, -1)    # (B, 1, D)
        x   = torch.cat([cls, patches], dim=1)    # (B, n_patches+1, D)
        x   = self.dropout(x + self.pos_emb)
        
        # Transformer encoder
        x   = self.transformer(x)
        x   = self.norm(x[:, 0])  # [CLS] token output
        x   = self.proj(x)
        
        return F.normalize(x, dim=-1)  # L2 normalise


# ---------------------------------------------------------------------------
# 2. Text Encoder
# ---------------------------------------------------------------------------

class TextEncoder(nn.Module):
    """
    Transformer text encoder for CLIP.
    
    Tokenises text and encodes via a standard Transformer encoder.
    The [EOS] token output is used as the global text representation,
    similar to how [CLS] is used in ViT.
    """
    def __init__(
        self,
        vocab_size:    int = 1000,
        embedding_dim: int = 256,
        n_heads:       int = 4,
        n_layers:      int = 4,
        max_seq_len:   int = 77,   # Same as original CLIP
        dropout:       float = 0.1,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embedding_dim)
        self.pos_emb   = nn.Embedding(max_seq_len, embedding_dim)
        self.dropout   = nn.Dropout(dropout)
        
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=n_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm        = nn.LayerNorm(embedding_dim)
        self.proj        = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: token indices (B, seq_len)
        Returns:
            text embeddings (B, embedding_dim), L2 normalised
        """
        B, T = input_ids.shape
        pos  = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x    = self.dropout(self.token_emb(input_ids) + self.pos_emb(pos))
        x    = self.transformer(x)
        x    = self.norm(x[:, -1])  # Last token = EOS position
        x    = self.proj(x)
        return F.normalize(x, dim=-1)


# ---------------------------------------------------------------------------
# 3. CLIP Model
# ---------------------------------------------------------------------------

class CLIP(nn.Module):
    """
    CLIP: Contrastive Language-Image Pre-Training.
    
    The temperature τ controls the sharpness of the contrastive distribution.
    Small τ → sharper distribution → model is more confident about which pairs match.
    Large τ → softer distribution → more tolerance for near-matches.
    Original CLIP paper found τ~0.07 works well; it's a learnable parameter.
    """
    def __init__(
        self,
        image_encoder: nn.Module,
        text_encoder:  nn.Module,
        temperature:   float = 0.07,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder  = text_encoder
        # Temperature as a learnable parameter
        self.log_temp = nn.Parameter(torch.tensor(math.log(temperature)))
    
    def forward(
        self,
        images:    torch.Tensor,      # (B, C, H, W)
        input_ids: torch.Tensor,      # (B, seq_len)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            image_embs: (B, D) normalised image embeddings
            text_embs:  (B, D) normalised text embeddings
            logits:     (B, B) similarity matrix
        """
        image_embs = self.image_encoder(images)
        text_embs  = self.text_encoder(input_ids)
        
        # Scale by temperature
        scale  = self.log_temp.exp()
        logits = (image_embs @ text_embs.T) * scale
        
        return image_embs, text_embs, logits
    
    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """InfoNCE loss: symmetric cross-entropy on the similarity matrix."""
        B      = logits.shape[0]
        labels = torch.arange(B, device=logits.device)
        loss_i = F.cross_entropy(logits, labels)      # Image → text direction
        loss_t = F.cross_entropy(logits.T, labels)    # Text → image direction
        return (loss_i + loss_t) / 2
    
    @torch.no_grad()
    def zero_shot_classify(
        self,
        image:         torch.Tensor,       # (1, C, H, W)
        class_prompts: torch.Tensor,       # (n_classes, seq_len) token IDs
    ) -> Tuple[int, torch.Tensor]:
        """
        Classify an image zero-shot given text descriptions of each class.
        
        Returns:
            predicted class index and probability distribution
        """
        self.eval()
        image_emb  = self.image_encoder(image)                      # (1, D)
        class_embs = self.text_encoder(class_prompts)               # (n_classes, D)
        
        similarities = (image_emb @ class_embs.T).squeeze()        # (n_classes,)
        probs        = F.softmax(similarities / self.log_temp.exp(), dim=0)
        predicted    = int(probs.argmax().item())
        
        return predicted, probs


# ---------------------------------------------------------------------------
# 4. Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("CLIP from Scratch — Demo")
    print("=" * 65)
    
    torch.manual_seed(42)
    
    # Build CLIP
    print("\n[1/4] Building CLIP model...")
    image_enc = ViTEncoder(image_size=32, patch_size=8, embedding_dim=128, n_layers=2)
    text_enc  = TextEncoder(vocab_size=500, embedding_dim=128, n_layers=2)
    clip      = CLIP(image_enc, text_enc)
    
    img_params   = sum(p.numel() for p in image_enc.parameters())
    text_params  = sum(p.numel() for p in text_enc.parameters())
    print(f"  Image encoder: {img_params:,} params")
    print(f"  Text encoder:  {text_params:,} params")
    
    # Forward pass
    print("\n[2/4] Forward pass...")
    B = 8
    images     = torch.randn(B, 3, 32, 32)
    input_ids  = torch.randint(0, 500, (B, 20))
    
    img_embs, txt_embs, logits = clip(images, input_ids)
    loss = clip.contrastive_loss(logits)
    
    print(f"  Image embeddings: {img_embs.shape}")
    print(f"  Text embeddings:  {txt_embs.shape}")
    print(f"  Logits matrix:    {logits.shape}")
    print(f"  Initial loss:     {loss.item():.4f}")
    
    # Verify L2 normalisation
    norms = img_embs.norm(dim=-1)
    print(f"  Image embedding norms (should all ≈1): {norms.tolist()[:3]}")
    
    # Quick training
    print("\n[3/4] Training for 20 steps...")
    optimizer = torch.optim.Adam(clip.parameters(), lr=1e-3)
    for step in range(20):
        images    = torch.randn(16, 3, 32, 32)
        input_ids = torch.randint(0, 500, (16, 20))
        _, _, logits = clip(images, input_ids)
        loss = clip.contrastive_loss(logits)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 5 == 0:
            print(f"  Step {step:2d}: loss={loss.item():.4f}  temp={clip.log_temp.exp().item():.4f}")
    
    # Zero-shot classification demo
    print("\n[4/4] Zero-shot classification demo...")
    test_image   = torch.randn(1, 3, 32, 32)
    n_classes    = 5
    class_tokens = torch.randint(0, 500, (n_classes, 20))
    class_names  = ["cat", "dog", "car", "ship", "plane"]
    
    pred_idx, probs = clip.zero_shot_classify(test_image, class_tokens)
    print(f"  Classes: {class_names}")
    print(f"  Probabilities: {[f'{p:.3f}' for p in probs.tolist()]}")
    print(f"  Predicted: '{class_names[pred_idx]}'")
    
    print("\n" + "=" * 65)
    print("✅ CLIP implementation complete!")


# ============================================================
# SECTION 4 & 5 — Deep Dive & Interview Prep
# ============================================================
"""
WEEK 7: MULTIMODAL MODELS — Deep Dive & Interview Prep

CONCEPTS:

1. CONTRASTIVE LEARNING (CLIP)
Analogy: A language immersion program where you're shown objects while hearing their names.
After seeing enough (image, text) pairs, you learn to associate visual concepts with
linguistic descriptions. CLIP does this at scale with 400M pairs from the internet.
The contrastive loss pushes matching pairs toward cosine similarity=1 and non-matching
pairs toward 0. The temperature τ controls the confidence of these assignments.

2. VISION TRANSFORMER (ViT)
Analogy: Reading comprehension, but the "words" are image patches. A 224×224 image
is split into 196 16×16 pixel patches, each linearly projected to a vector —
becoming a sequence of "visual tokens." The Transformer encoder then computes
self-attention across all 196 patch positions simultaneously.
The [CLS] token learns to aggregate global information across all patch interactions.

3. CROSS-MODAL ALIGNMENT
After CLIP training, "a photograph of a dog" and an image of a dog produce nearby
embeddings. This alignment enables: (1) zero-shot classification without training
examples; (2) image search by text query; (3) cross-modal retrieval;
(4) initialising vision-language models (LLaVA, Flamingo) by plugging CLIP
as the image encoder into an LLM.

4. WHISPER ARCHITECTURE
Whisper converts audio to log-mel spectrogram (frequency × time representation),
then encodes it with a CNN + Transformer encoder. A Transformer decoder auto-
regressively generates transcription tokens. This is sequence-to-sequence translation:
the encoder speaks "audio," the decoder produces "text." Weak supervision at scale
(YouTube subtitles, podcasts) enables robustness to accents, noise, and domains.

5. VISION-LANGUAGE MODELS (LLaVA, Flamingo)
Modern VLMs connect a visual encoder (CLIP ViT) to a language model via a projection
layer. The projection maps image embeddings into the LLM's embedding space.
During fine-tuning, only the projection (and optionally the LLM via LoRA) is trained;
the CLIP encoder is frozen. This is why strong visual encoders are critical: they
directly determine how much visual information can be communicated to the LLM.

MISCONCEPTIONS:
1. "CLIP embeds images and text in the same absolute space, so distances are meaningful."
CLIP embeddings are aligned only in *direction* (cosine similarity), not magnitude.
You cannot meaningfully compare |image_emb - text_emb| with Euclidean distance.
All comparisons should use cosine similarity (or dot product after L2 normalisation).

2. "ViT needs more data than CNNs to work well." True for ViT-Base trained from scratch.
However, when pretrained on large datasets (JFT-300M, ImageNet-21k, CLIP), ViT
outperforms CNNs even on small downstream tasks. Data-efficient ViT variants (DeiT, EfficientViT)
close the data efficiency gap significantly.

3. "Whisper is a diffusion model for audio." No — Whisper is a discriminative seq2seq
model. Diffusion-based audio models (like Stable Audio or AudioLM) generate new audio.
Whisper *transcribes* existing audio to text. Different task, different architecture.

INTERVIEW QUESTIONS (5):

Q1: How does CLIP enable zero-shot classification?
A: After CLIP training, text and image embeddings live in an aligned space. For zero-shot
classification of N classes: embed N text prompts ("a photo of a {class}") to get
N text embeddings. Embed the query image. The predicted class = text embedding with
highest cosine similarity to the image embedding. No training examples are needed —
the model generalises from the semantic similarity learned during pretraining.

Q2: What is the [CLS] token in ViT and why is it used?
A: [CLS] is a learnable vector prepended to the sequence of patch embeddings. Unlike
patch tokens (which have spatial meaning), [CLS] has no fixed spatial role — its
value after all attention layers is determined entirely by what it attends to.
This makes it an ideal global "summary" of the image. Mean pooling of all patch
outputs is an alternative that also works well in practice.

Q3: Why does CLIP use symmetric loss (both image-to-text and text-to-image directions)?
A: The contrastive loss can be computed row-wise (for each image, rank all texts) or
column-wise (for each text, rank all images). Using both directions doubles the effective
learning signal per batch and prevents the model from being good at retrieval in only
one direction. For a retrieval system that needs to find both "image given text" AND
"text given image," symmetric training is critical.

Q4: What is the log-mel spectrogram and why is it used as Whisper's input?
A: A log-mel spectrogram is a 2D representation of audio (mel frequency bands × time
frames). "Mel" scaling approximates human auditory perception — frequency resolution
is higher at low frequencies, lower at high frequencies. "Log" compresses the dynamic
range (matching how humans perceive loudness). The result is a compact, perceptually
meaningful 2D tensor that Whisper's CNN+Transformer can process analogously to how a
ViT processes image patches.

Q5: What is the projection layer in LLaVA and why does it need to be trained?
A: LLaVA connects a CLIP image encoder to an LLM via a linear projection (or MLP).
The projection maps CLIP embeddings (e.g., 768-dim) into the LLM's token embedding
space (e.g., 4096-dim). Without the projection, the image embeddings are in CLIP's
learned space — semantically rich but not in the format the LLM expects.
The projection must be trained so the LLM can "read" image information, but the CLIP
encoder and LLM can remain frozen (or fine-tuned with LoRA for better performance).

SYSTEM DESIGN: Design a visual search system for an e-commerce platform.
Index: Extract CLIP image embeddings for all product images (offline job, batch GPU).
Store embeddings in a vector database (Pinecone, Qdrant) with product metadata.
Query: User uploads a photo → extract CLIP image embedding at request time →
ANN search → retrieve top 50 → re-rank with a cross-encoder (image+text product title).
Business rules: filter by category, price range, stock availability.
Serving: GPU instance for embedding extraction; FAISS on distributed CPU for ANN.
Latency target: < 200ms p99 for search (embedding: 50ms, ANN: 20ms, re-rank: 100ms).
Monitoring: search CTR, add-to-cart rate, embedding drift (monthly re-index).
"""
