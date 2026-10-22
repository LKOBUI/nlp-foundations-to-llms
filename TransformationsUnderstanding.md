# Transformer Understanding: From TF-IDF to Encoder--Decoder Architecture

> Consolidated learning notes from our discussions. The original
> questions are retained as section prompts, and the explanations are
> arranged in a learning sequence. The contents below link to the
> corresponding sections.

## Contents

1.  [TF-IDF: Importance of Words](#1-tf-idf-importance-of-words)
2.  [Why Word Embeddings Are Needed](#2-why-word-embeddings-are-needed)
    -   [Example 1: TF-IDF and
        synonyms](#example-1--tf-idf-cannot-inherently-understand-synonyms)
    -   [Example 2: Word embeddings and cosine
        similarity](#example-2--how-word-embeddings-represent-meaning)
3.  [Statistical Embeddings, Neural Embeddings, PCA, and
    Top-K](#3-statistical-embeddings-neural-embeddings-pca-and-top-k)
4.  [Topic Modeling and Vectors](#4-topic-modeling-and-vectors)
5.  [DistilBERT: Tokenization, Warnings, and Contextual
    Embeddings](#5-distilbert-tokenization-warnings-and-contextual-embeddings)
6.  [Padding IDs and Attention
    Masks](#6-padding-ids-and-attention-masks)
7.  [Contextual Embeddings vs. Traditional Word
    Embeddings](#7-contextual-embeddings-vs-traditional-word-embeddings)
8.  [Transformer Input Architecture](#8-transformer-input-architecture)
9.  [How Attention Weights Are
    Learned](#9-how-attention-weights-are-learned)
10. [Encoder (N×): Multi-Head Attention and Final
    Output](#10-encoder-n-multi-head-attention-and-final-output)
11. [Multi-Head Attention Compared with CNN
    Filters](#11-multi-head-attention-compared-with-cnn-filters)
12. [Decoder: Shifted-Right Input, Cross-Attention, and
    Softmax](#12-decoder-shifted-right-input-cross-attention-and-softmax)
13. [Quick Revision Summary](#13-quick-revision-summary)

------------------------------------------------------------------------

## 1. TF-IDF: Importance of Words

**Discussion prompt:** Arrange a step-by-step TF-IDF formula example
without changing the example method or actual example.

TF-IDF (Term Frequency--Inverse Document Frequency) estimates how
important a word is to a document relative to a corpus.

$$
\\operatorname{TF\\text{-}IDF}(t,d)
=
\\operatorname{TF}(t,d)\\times\\operatorname{IDF}(t)
$$

Using the smoothed formula discussed:

$$
\\operatorname{TF}(t,d)=
\\frac{\\text{count of }t\\text{ in }d}{\\text{total terms in }d}
$$

$$
\\operatorname{IDF}(t)=
\\ln\\left(\\frac{N+1}{df(t)+1}\\right)
$$

Here, $N$ is the total number of documents and $df(t)$ is the number of
documents containing term $t$.

### Example corpus

  Document   Text
  ---------- -------------------------
  D1         the cat sat on the mat
  D2         the dog ran in the park
  D3         the cat chased the dog
  D4         birds fly in the sky

There are four documents, so $N=4$. For `cat` in D1, the term appears
once in a six-term document, and two documents contain `cat`.

$$
\\operatorname{TF}(\\text{cat},D1)=\\frac{1}{6}\\approx0.1667
$$

$$
\\operatorname{IDF}(\\text{cat})=
\\ln\\left(\\frac{4+1}{2+1}\\right)
=\\ln\\left(\\frac{5}{3}\\right)\\approx0.5108
$$

$$
\\operatorname{TF\\text{-}IDF}(\\text{cat},D1)
\\approx0.1667\\times0.5108\\approx0.0852
$$

For `the`, which appears in all four documents:

$$
\\operatorname{IDF}(\\text{the})=\\ln(5/5)=0
$$

Its TF-IDF is zero under this formula. For `mat`, which occurs only in
D1:

$$
\\operatorname{IDF}(\\text{mat})=\\ln(5/2)\\approx0.9163
$$

$$
\\operatorname{TF\\text{-}IDF}(\\text{mat},D1)
\\approx\\frac{1}{6}\\times0.9163\\approx0.1527
$$

**Takeaway:** TF-IDF emphasizes words frequent in a document but
uncommon across the corpus. It does not inherently understand synonyms
or context.

------------------------------------------------------------------------

## 2. Why Word Embeddings Are Needed

**Original question:** "We have TF-IDF why we required word embedding,
explain with example"

TF-IDF and word embeddings both convert text into numbers, but they
encode different information.

-   **TF-IDF:** How important is a word in a document?
-   **Word embeddings:** What relationships or semantic similarities can
    be learned between words?
-   **Contextual embeddings:** How is a token being used in this
    particular sentence?

### Example 1 --- TF-IDF cannot inherently understand synonyms

  Document   Text
  ---------- ------------------------
  D1         The car is fast
  D2         The automobile is fast
  D3         The bus is fast

Using the simplified vocabulary `[car, automobile, bus, fast]`, an
illustrative TF-IDF matrix could look like this:

  Feature         D1    D2    D3
  ------------ ----- ----- -----
  car            0.5     0     0
  automobile       0   0.5     0
  bus              0     0   0.5
  fast           0.2   0.2   0.2

*These values are illustrative, not actual TF-IDF calculations.*

`car` and `automobile` occupy separate dimensions. In this simplified
example, D1 and D2 have no shared nonzero features, so their cosine
similarity is zero. Humans recognize that the sentences are similar, but
TF-IDF does not inherently know that `car` and `automobile` are
synonyms.

### Example 2 --- How word embeddings represent meaning

Imagine an embedding model produces these illustrative vectors:

  Word           Dimension 1   Dimension 2
  ------------ ------------- -------------
  car                   0.90          0.80
  automobile            0.88          0.82
  bus                   0.85          0.30
  banana                0.10          0.90

These are example values, not actual pretrained embeddings. Dimensions
do not generally have fixed human-readable meanings.

Cosine similarity compares vector directions:

$$
\\operatorname{cosine}(A,B)=\\frac{A\\cdot B}{\\|A\\|\\|B\\|}
$$

For `car` and `automobile`:

$$
A=(0.90,0.80),\\quad B=(0.88,0.82)
$$

$$
A\\cdot B=(0.90)(0.88)+(0.80)(0.82)=1.448
$$

$$
\\operatorname{cosine}(A,B)=
\\frac{1.448}{\\sqrt{1.45}\\sqrt{1.4468}}\\approx0.9999
$$

For `car` and `banana`, where $C=(0.10,0.90)$:

$$
\\operatorname{cosine}(A,C)=
\\frac{0.81}{\\sqrt{1.45}\\sqrt{0.82}}\\approx0.743
$$

These values only demonstrate the calculation; actual scores depend on
the embedding model.

**Takeaway:** TF-IDF represents word importance and occurrence.
Embeddings can represent semantic relationships learned from text.

------------------------------------------------------------------------

## 3. Statistical Embeddings, Neural Embeddings, PCA, and Top-K

**Original question:** The user understood embeddings as coming from
statistical methods such as SVD and pretrained neural models, and asked
whether Top-K selection is similar to PCA component selection.

Statistical/matrix-factorization approaches can use co-occurrence
matrices and SVD, as in Latent Semantic Analysis (LSA). Traditional
neural embedding methods include Word2Vec. Contextual neural models
include BERT, DistilBERT, and GPT.

### PCA explained variance

In PCA, eigenvalues are ordered from largest to smallest. The explained
variance ratio is:

$$
\\operatorname{ExplainedVarianceRatio}_i=
\\frac{\\lambda_i}{\\sum_{j=1}^{n}\\lambda_j}
$$

The cumulative sum can be used to retain, for example, 95% of the
variance. This does **not** mean 95% prediction accuracy.

### How Top-K differs

Top-K means retaining the $K$ highest-ranked items under a specified
score:

-   **PCA:** Keep principal components with the largest eigenvalues to
    retain variance.
-   **LLM Top-K sampling:** Keep the $K$ candidate next tokens with the
    highest predicted probabilities, then sample from that restricted
    set.

They share a general selection principle but are not the same operation.

------------------------------------------------------------------------

## 4. Topic Modeling and Vectors

**Original question:** "what topic modeling. Can we achive it by vector"

Yes. Topic modeling can use numerical vectors, but vectorization alone
does not perform topic modeling. A method must discover groups or
patterns.

Topic modeling discovers recurring themes in documents, such as
technology, sports, finance, or health.

### Example documents

  Document   Text
  ---------- ------------------------------------------------
  D1         Neural networks train deep learning models
  D2         AI models learn using neural networks
  D3         Cricket players win sports tournaments
  D4         Football players compete in sports tournaments

A method might discover two groups: D1/D2 (AI and machine learning) and
D3/D4 (sports). The algorithm does not initially know the human-readable
labels; they are assigned after examining the discovered patterns.

  -----------------------------------------------------------------------
  Method                  Typical input           How it discovers topics
  ----------------------- ----------------------- -----------------------
  LDA                     Word-count matrix       Learns probabilistic
                                                  topic-word and
                                                  document-topic
                                                  distributions.

  NMF                     Often TF-IDF            Factorizes a
                                                  non-negative
                                                  document-term matrix
                                                  into topic-related
                                                  patterns.

  K-means                 TF-IDF or embeddings    Groups similar document
                                                  vectors; clusters can
                                                  be interpreted as
                                                  candidate topics.

  BERTopic                Document embeddings     Clusters semantic
                                                  representations and
                                                  uses class-based TF-IDF
                                                  to describe topics.
  -----------------------------------------------------------------------

### Using DistilBERT for topic modeling

1.  Prepare documents.
2.  Generate token representations using DistilBERT.
3.  Pool token representations into one vector per document.
4.  Cluster document vectors using K-means or HDBSCAN.
5.  Extract representative keywords using TF-IDF or class-based TF-IDF.
6.  Inspect and label the discovered topics.

PCA or SVD can reduce dimensionality, but dimensionality reduction alone
does not discover or label topics.

------------------------------------------------------------------------

## 5. DistilBERT: Tokenization, Warnings, and Contextual Embeddings

**Discussion prompt:** The user shared a `DistilBertModel LOAD REPORT`
containing `UNEXPECTED` weights and this token list:

`['[CLS]', 'the', 'bank', 'of', 'the', 'river', 'was', 'flooded', '[SEP]']`

### Understanding the `UNEXPECTED` weights warning

A checkpoint may contain weights for a masked-language-modeling head,
such as `vocab_transform`, `vocab_layer_norm`, and `vocab_projector`.
When loaded with `DistilBertModel`, only the base Transformer encoder is
required. The extra prediction-head weights are unused and can be
reported as `UNEXPECTED`.

-   `UNEXPECTED`: The checkpoint has weights that the requested
    architecture does not use.
-   `MISSING`: The requested architecture expects weights that were not
    found. Investigate whether the difference is expected, such as a
    newly added task head, or indicates a mismatch.

For contextual embeddings, `AutoModel` or `DistilBertModel` is
appropriate; this warning is not a failure. If you need the
masked-language-modeling head, use `AutoModelForMaskedLM`.

### Complete DistilBERT contextual-embedding code

``` python
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

checkpoint = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)
model.eval()

sentence = "The bank of the river was flooded"

inputs = tokenizer(
    sentence,
    return_tensors="pt",
    padding=True,
    truncation=True,
    return_attention_mask=True
)

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

print("Original sentence:", sentence)
print("Tokenized list:", tokens)
print("Input IDs:", inputs["input_ids"])
print("Attention mask:", inputs["attention_mask"])
print("PAD token:", tokenizer.pad_token)
print("PAD token ID:", tokenizer.pad_token_id)

with torch.no_grad():
    outputs = model(**inputs)

embeddings = outputs.last_hidden_state
print("\\nEmbedding tensor shape:", embeddings.shape)
# For one sentence with 9 tokens: [1, 9, 768]

print("\\nContextual embeddings (first 3 features):")
for tok, emb in zip(tokens, embeddings[0]):
    print(f"{tok:12s}: {np.round(emb[:3].cpu().numpy(), 4)}")

bank_index = tokens.index("bank")
bank_embedding = embeddings[0, bank_index, :]

print("\\nBank embedding shape:", bank_embedding.shape)
print("First 10 features:", bank_embedding[:10].cpu().numpy())
```

For a sentence with nine tokens, the expected hidden-state shape is
`torch.Size([1, 9, 768])`. The 768 values are learned hidden features,
not 768 PCA components.

### If the matching masked-language-model head is required

``` python
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

checkpoint = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForMaskedLM.from_pretrained(
    checkpoint,
    output_hidden_states=True
)
model.eval()

sentence = "The bank of the river was flooded"
inputs = tokenizer(sentence, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

embeddings = outputs.hidden_states[-1]
print("Embedding shape:", embeddings.shape)
```

This loads the architecture matching the masked-language-modeling head.
The hidden states remain available for contextual-embedding inspection.

------------------------------------------------------------------------

## 6. Padding IDs and Attention Masks

**Original question:** "In Tokenization method NLP, Can I assume padding
always assignee by 0 & CREATING ATTENTION MASKS always 1. And 0 & 1
reserved for padding and CREATING ATTENTION MASKS."

No. Token IDs and attention masks are separate representations.

For DistilBERT:

-   Padding token: `[PAD]`
-   Padding token ID: `0`
-   Attention mask `1`: valid token position
-   Attention mask `0`: padding position

Token ID `1` is a vocabulary ID, not an attention-mask flag. Other
tokenizers can use a different padding ID, and some do not define a
dedicated padding token by default. Padding is only added when requested
or needed for batching.

``` python
print("PAD token:", tokenizer.pad_token)
print("PAD token ID:", tokenizer.pad_token_id)

inputs = tokenizer(
    ["The bank flooded", "The bank of the river was flooded"],
    padding=True,
    return_attention_mask=True,
    return_tensors="pt"
)

print("Input IDs:\\n", inputs["input_ids"])
print("Attention mask:\\n", inputs["attention_mask"])
```

------------------------------------------------------------------------

## 7. Contextual Embeddings vs. Traditional Word Embeddings

**Original question:** "What is Contextual Embeddings (BERT, GPT,
Transformer models) & how it is different from Word Embeddings?"

### Traditional word embeddings

Word2Vec and GloVe typically assign one fixed vector to each vocabulary
word. For example, `bank` generally gets the same vector in:

-   "I deposited money at the bank."
-   "The bank of the river was flooded."

The vector can encode broad associations learned from training, but it
does not change for each occurrence.

### Contextual embeddings

BERT, DistilBERT, and GPT produce contextual hidden representations. The
representation of `bank` can differ between financial and river contexts
because the model processes surrounding tokens.

$$
\\mathbf{h}_i=f_\\theta(x_1,x_2,\\ldots,x_n)_i
$$

Here, $x_i$ is a token, the sequence provides context, $f_\\theta$ is
the trained Transformer, and $\\mathbf{h}_i$ is the contextual
representation at position $i$.

  -----------------------------------------------------------------------
  Feature                 Traditional word        Contextual embeddings
                          embeddings              
  ----------------------- ----------------------- -----------------------
  Examples                Word2Vec, GloVe         BERT, DistilBERT, GPT
                                                  hidden states

  Representation          Usually one vector per  Depends on context
                          vocabulary word         

  Different senses of a   Limited handling        Better able to
  word                                            represent different
                                                  senses

  Typical use             Word similarity,        Contextual
                          lightweight NLP         classification,
                                                  semantic search
  -----------------------------------------------------------------------

A Transformer still has an initial token-embedding layer. Its
Transformer layers process those initial representations to produce
contextual hidden states.

### Inspecting the contextual vector for `bank`

``` python
import torch
from transformers import AutoTokenizer, AutoModel

checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)
model.eval()

sentences = [
    "I deposited money at the bank.",
    "The bank of the river was flooded."
]

with torch.no_grad():
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt")
        outputs = model(**inputs)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        bank_index = tokens.index("bank")
        bank_vector = outputs.last_hidden_state[0, bank_index, :]

        print("Sentence:", sentence)
        print("Bank vector shape:", bank_vector.shape)
        print("First five features:", bank_vector[:5])
```

The two vectors will generally differ because the surrounding contexts
differ.

------------------------------------------------------------------------

## 8. Transformer Input Architecture

**Original question:** The user understood input embeddings as
vectorization, imagined a matrix such as 50 × 768, and thought
sinusoidal positional encoding uses cosine mathematics to find word
relationships.

The understanding was partly correct. The key correction is:

-   Input embeddings convert token IDs into vectors.
-   Positional information represents token positions.
-   Self-attention learns token relationships.

### Token embeddings and dimensions

For `distilbert-base-uncased`, the vocabulary contains approximately
30,522 tokens, and each token embedding has 768 features. The embedding
table is approximately:

$$
E\\in\\mathbb{R}^{30522\\times768}
$$

If the input contains nine tokens, the embedding lookup returns:

$$
X_{\\text{embed}}\\in\\mathbb{R}^{9\\times768}
$$

If it contains 50 tokens:

$$
X_{\\text{embed}}\\in\\mathbb{R}^{50\\times768}
$$

Thus, 50 × 768 is the representation for a 50-token sequence, not the
full vocabulary table.

### Positional information

The original Transformer uses sinusoidal positional encodings:

$$
PE(pos,2i)=
\\sin\\left(\\frac{pos}{10000^{2i/d}}\\right)
$$

$$
PE(pos,2i+1)=
\\cos\\left(\\frac{pos}{10000^{2i/d}}\\right)
$$

These functions represent token positions; they do not directly
calculate semantic relationships. **DistilBERT uses learned positional
embeddings**, rather than the original Transformer's fixed sinusoidal
encoding.

Conceptually, in a BERT-style model:

$$
X_0=X_{\\text{token}}+X_{\\text{position}}
$$

DistilBERT also applies embedding-layer normalization and dropout after
combining its embeddings.

### Self-attention learns relationships

$$
Q=XW_Q,\\qquad K=XW_K,\\qquad V=XW_V
$$

$$
\\operatorname{Attention}(Q,K,V)=
\\operatorname{softmax}
\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right)V
$$

Queries and keys determine attention scores. Softmax normalizes those
scores, and the resulting weights combine value vectors.

**Corrected understanding:** Input embeddings supply initial vectors;
positional embeddings supply position information; self-attention learns
relationships between tokens.

------------------------------------------------------------------------

## 9. How Attention Weights Are Learned

**Original question:** The user asked how $W_Q$, $W_K$, and $W_V$ are
updated, given knowledge of Glorot/He initialization, trainable
parameters, FFNs, and backpropagation.

Transformers are neural networks. They use initialization, forward
propagation, a loss function, backpropagation, and an optimizer.

### Initialization

When training from scratch, projection matrices can use an appropriate
initialization scheme:

``` python
import torch
import torch.nn as nn

d_model = 768
d_k = 64

W_Q = nn.Parameter(torch.empty(d_model, d_k))
W_K = nn.Parameter(torch.empty(d_model, d_k))
W_V = nn.Parameter(torch.empty(d_model, d_k))

nn.init.xavier_uniform_(W_Q)
nn.init.xavier_uniform_(W_K)
nn.init.xavier_uniform_(W_V)
```

These are illustrative standalone parameters. Real Transformers normally
define projections inside attention modules or linear layers, often with
biases. Initialization can use Xavier/Glorot, normal initialization, or
another model-specific scheme. A pretrained model loads its learned
weights from a checkpoint.

### Forward pass and attention scores

$$
Q=XW_Q,\\quad K=XW_K,\\quad V=XW_V
$$

$$
A=\\operatorname{softmax}
\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right),\\quad Z=AV
$$

$A$ is computed dynamically from the current input. It is not normally a
persistent parameter directly updated by the optimizer. The optimizer
updates the projection matrices, which indirectly influence attention
patterns.

### Backpropagation and optimization

A training objective produces loss $L$. Backpropagation calculates
gradients:

$$
\\frac{\\partial L}{\\partial W_Q},\\quad
\\frac{\\partial L}{\\partial W_K},\\quad
\\frac{\\partial L}{\\partial W_V}
$$

A basic gradient-descent update is:

$$
W_Q^{\\text{new}}=W_Q^{\\text{old}}-
\\eta\\frac{\\partial L}{\\partial W_Q}
$$

where $\\eta$ is the learning rate. Transformers commonly use AdamW or
another optimizer. The gradients propagate through the whole
computational graph, including attention, feed-forward weights,
embeddings, and other trainable parameters. The feed-forward network
does not independently update attention weights; all trainable
parameters are optimized through the shared loss.

  -----------------------------------------------------------------------
  Scenario                            What happens
  ----------------------------------- -----------------------------------
  Training from scratch               Parameters start from
                                      initialization and are learned
                                      through optimization.

  Pretrained-model inference          Checkpoint weights are loaded and
                                      ordinarily remain unchanged.

  Fine-tuning                         Existing weights are further
                                      optimized unless frozen.

  LoRA                                Typically freezes base weights and
                                      optimizes additional low-rank
                                      parameters.
  -----------------------------------------------------------------------

Using `torch.no_grad()` for inference does not calculate gradients or
update parameters.

------------------------------------------------------------------------

## 10. Encoder (N×): Multi-Head Attention and Final Output

**Original question:** "Focus on Encode (Nx) Sections from transformer
architecture. You have mentions Multi-Head Attentions. First tell me why
its call Multi-head we need to just calculate attentions score? What is
final outcome outcome from encoder(Nx)"

### Why Multi-Head Attention?

One head calculates relationships between tokens using query, key, and
value projections:

$$
Q_h=XW_Q^{(h)},\\quad K_h=XW_K^{(h)},\\quad V_h=XW_V^{(h)}
$$

$$
\\operatorname{Head}_h=
\\operatorname{softmax}\\left(
\\frac{Q_hK_h^\\top}{\\sqrt{d_k}}
\\right)V_h
$$

Attention involves three steps:

1.  Calculate query-key similarity scores.
2.  Apply softmax to obtain attention weights.
3.  Combine value vectors using those weights.

Multiple heads allow the model to learn different attention patterns in
parallel. One head may learn pronoun-related patterns, while another may
learn action/object relationships. These are learned patterns, not
manually fixed assignments.

$$
\\operatorname{MultiHead}(X)=
\\operatorname{Concat}(\\operatorname{Head}_1,\\ldots,\\operatorname{Head}_H)W_O
$$

For the original Transformer with $d_{\\text{model}}=768$ and 12 heads,
each head can use $d_k=d_v=768/12=64$. Their outputs concatenate to 768
features per token before the final projection.

### What does Encoder (N×) mean?

The encoder layer is repeated $N$ times. Each layer contains:

1.  Multi-head self-attention.
2.  Residual connection and layer normalization.
3.  Feed-forward neural network.
4.  Another residual connection and layer normalization.

The output of one layer becomes the input to the next:

$$
X_0\\rightarrow X_1\\rightarrow\\cdots\\rightarrow X_N
$$

The final output is:

$$
X_N\\in\\mathbb{R}^{L\\times d_{\\text{model}}}
$$

For a batch, its shape is $(B,L,d_{\\text{model}})$. For one DistilBERT
sentence with nine tokens, the expected shape is:

``` text
torch.Size([1, 9, 768])
```

DistilBERT has six Transformer encoder layers and 12 attention heads per
layer. The output is **not merely an attention-score matrix**: it is a
sequence of contextual token representations. The representation for
`bank` can incorporate information from `river` and `flooded`.

### Does the encoder produce one sentence vector?

Not automatically. It produces one vector per token position. To create
a document vector, apply an additional operation such as mean pooling, a
suitable `[CLS]` representation, or task-specific pooling.

------------------------------------------------------------------------

## 11. Multi-Head Attention Compared with CNN Filters

**Original question:** "So we can multi head attentions just like CNN
architecture where we use filter which represent number of feature which
break image into very small part. Just give me you thought, it is
correct or not. Do not expend answer keep short as possible?"

**Answer: Partially correct.**

-   **CNN filters:** Learn local spatial patterns such as edges,
    textures, and shapes by scanning small image regions.
-   **Multi-head attention:** Learns different relationships between
    tokens, potentially connecting distant tokens across a sequence.

**Similarity:** Multiple CNN filters learn different visual features;
multiple attention heads learn different token relationships.

**Key difference:** CNN filters focus on local regions, whereas
self-attention can relate distant tokens directly.

------------------------------------------------------------------------

## 12. Decoder: Shifted-Right Input, Cross-Attention, and Softmax

**Original question:** "Now focus on Right part of the transformer
architecture where the final outcome id Output probability from SoftMax.
also called decoder. First tell me where the output (Shifter right) come
from? Is it feed back from SoftMax out put. Why it is called decoder?
From encoder architecture it's not much different. keep answer details.
Do not sort answer."

### Where does the shifted-right input come from?

In the original encoder--decoder Transformer, the decoder receives the
target sequence shifted one position to the right during training. This
is called **teacher forcing**.

Suppose the desired target is:

`[BOS] J'aime les chats [EOS]`

The decoder input is:

`[BOS] J'aime les chats`

The target tokens are:

`J'aime les chats [EOS]`

The model learns to predict the next token at each position. A causal
mask prevents it from seeing future target tokens.

### Does the Softmax output feed back into the decoder?

It depends on training versus inference.

**During training:** The shifted-right target sequence is supplied to
the decoder. The decoder produces hidden representations, and Linear +
Softmax produces next-token probabilities. Predictions are compared with
correct next tokens to calculate loss, and backpropagation updates the
trainable parameters. The predicted token is not normally fed back as
the next decoder input during standard teacher-forced training.

**During inference:** The target sentence is not known. The decoder
predicts a next-token distribution, a token is selected and appended to
the current sequence, and the model predicts the next token.

Example:

1.  Current input: `[BOS] I love`
2.  Model predicts next-token probabilities.
3.  Selected token: `cats`
4.  New input: `[BOS] I love cats`
5.  Repeat until the sequence ends or a stopping criterion is reached.

The selected token's **token ID** becomes part of the next decoder input
and is converted to an embedding. The full Softmax probability vector is
not ordinarily fed back as the next token input.

### Why is it called a decoder?

In the original Transformer:

-   **Encoder:** Processes the source sequence into contextual
    representations.
-   **Decoder:** Uses those source representations and previously
    generated target tokens to produce the target sequence.

For translation, the encoder processes `I love cats`, and the decoder
generates `J'aime les chats`. The decoder transforms encoded source
information and target-side context into output tokens; it does not
simply reverse the encoder's calculations.

### Why is the decoder similar to the encoder?

Both use attention, feed-forward layers, residual connections, and
normalization. The original Transformer decoder has additional behavior:

1.  **Masked self-attention:** Each target position can attend only to
    preceding and current target tokens.
2.  **Cross-attention:** Queries come from the decoder; keys and values
    come from encoder output.
3.  **Output projection:** A Linear layer maps decoder hidden states to
    vocabulary logits, and Softmax converts them into probabilities.

### Encoder versus decoder

  -----------------------------------------------------------------------
  Encoder                             Decoder (original Transformer)
  ----------------------------------- -----------------------------------
  Receives source-token embeddings    Receives shifted-right target-token
                                      embeddings

  Uses self-attention over valid      Uses causal/masked self-attention
  source positions                    

  Produces source contextual          Uses target context to predict
  representations                     output tokens

  No mandatory vocabulary Softmax     Final Linear + Softmax produces
  output                              next-token probabilities

  Repeated encoder layers             Repeated decoder layers, including
                                      cross-attention
  -----------------------------------------------------------------------

The original Transformer decoder contains two different attention
mechanisms: masked self-attention over target tokens and cross-attention
over encoder output.

### What comes out before Softmax?

The decoder first produces hidden representations. Suppose:

$$
H_{\\text{decoder}}\\in\\mathbb{R}^{L\\times512}
$$

A learned Linear layer maps hidden features to vocabulary logits:

$$
Z=H_{\\text{decoder}}W_{\\text{vocab}}+b
$$

If the vocabulary has 30,000 tokens, each position has 30,000 logits.
Softmax converts them into next-token probabilities:

$$
P(t_i\\mid t_{<i},X)=\\operatorname{softmax}(Z_i)
$$

Here, $t_{<i}$ denotes preceding target tokens, and $X$ denotes the
encoded source sequence in an encoder--decoder model.

### Original Transformer vs. BERT and GPT

  -----------------------------------------------------------------------
  Model                   Architecture            Processing
  ----------------------- ----------------------- -----------------------
  Original Transformer    Encoder + decoder       Decoder uses encoder
                                                  output through
                                                  cross-attention.

  BERT / DistilBERT       Encoder-only            Produces contextual
                                                  representations; not a
                                                  standard autoregressive
                                                  generator.

  GPT                     Decoder-only            Uses causal
                                                  self-attention to
                                                  predict next tokens; no
                                                  separate encoder block.
  -----------------------------------------------------------------------

**Final understanding:** The decoder receives shifted-right target
tokens during training and previously generated tokens during inference.
It uses masked self-attention for target context and, in the original
encoder--decoder Transformer, cross-attention to access encoder
information. Its final hidden representations are mapped to vocabulary
logits and then probabilities. During generation, the selected token is
appended to the input and the process repeats.

------------------------------------------------------------------------

## 13. Quick Revision Summary

-   **TF-IDF** measures word importance using within-document frequency
    and corpus-level rarity.
-   **Traditional embeddings** represent words with learned vectors;
    contextual embeddings depend on surrounding context.
-   **SVD/PCA** can reduce dimensionality or reveal matrix structure,
    but they are not synonymous with neural embedding training.
-   **Topic modeling** discovers recurring themes; vectors are
    representations used by a topic-discovery method.
-   **Token IDs** and **attention masks** are different: a tokenizer's
    PAD ID is not the same as a mask value.
-   **Input embeddings** map token IDs to vectors; positional embeddings
    provide position information.
-   **Self-attention** learns token relationships through query-key
    scores and value aggregation.
-   **Multi-head attention** learns several attention patterns in
    parallel.
-   **Encoder (N×)** outputs contextual representations for every input
    token.
-   **Attention matrices** are computed during the forward pass; $W_Q$,
    $W_K$, and $W_V$ are learnable parameters.
-   **Backpropagation and an optimizer** update trainable Transformer
    parameters during training or fine-tuning.
-   **Decoder shifted-right input** comes from the target sequence
    during teacher-forced training; during inference, generated tokens
    are appended and fed back.
-   **The original Transformer decoder** includes masked self-attention,
    cross-attention, and an output projection to vocabulary
    probabilities.
-   **GPT** is decoder-only; **BERT/DistilBERT** are encoder-only.
