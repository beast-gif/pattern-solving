# Similarity Pipeline: What Do Two Things Have in Common?

An agent pipeline that reads a dataset (for example, a PDF transcript of a leader's interview), breaks it into individual ideas, scores every idea on a shared set of features, and finds **which ideas are most alike and what exactly they have in common**.

It uses the same core idea as Spotify's autoplay and Smart Shuffle: turn every item into a vector, then compare vectors with cosine similarity. On top of that, each similar pair gets an explanation of *which features* make it similar.

**Core rule:** the LLM *describes* the items, and code does *all* the math. No similarity number ever comes from an LLM.

---

## Architecture

```
PDF ──► [1] Items ──► [2] Schema ──► [3] Sizing ──► [4] Profilers ──► [5] Compare ──► [6] Explainer
        (code)        (1 Sonnet      (code)         (W Haiku agents   (PyTorch,       (1 Sonnet
                       agent + you)                  in parallel)      no LLM)         agent)
```

| Stage | Who does it | Input | Output |
|---|---|---|---|
| 1. Items | Code (pdfplumber + cleaning) | PDF | `items.json` |
| 2. Schema | 1 deep agent (Sonnet) + your review | ~20 sample items | `schema.json` |
| 3. Sizing | Code | Item lengths, context window | k (items per agent), W (agents at once) |
| 4. Profilers | W deep agents in parallel (Haiku) | Items + frozen schema | `profiles.json` |
| 5. Compare | PyTorch | Profiles | Top pairs, contributions, heatmap |
| 6. Explainer | 1 deep agent (Sonnet) | Computed numbers + evidence | `report.md` |

---

## Stage by stage

### 1. Items (code)

The PDF is extracted with pdfplumber, then cleaned: broken lines and page breaks are joined, noise is removed (headers, timestamps, "(laughs)", filler words), and speech-to-text caption errors are fixed. The text is then split into **one item per idea**.

```json
{"id": 3, "title": "Step 2: delete the part or process", "text": "And then the second thing is try to delete..."}
```

Items should be short, ideally under about 500 words. Long items that mix several ideas get hedged scores (see *Known limitations*).

### 2. Schema (1 Sonnet agent + your review)

The **schema** is the fixed list of features every item is scored on, from 0 to 1. Sonnet reads a sample of items and proposes 5–8 features. Each feature has a description and anchors that define 0, 0.5 and 1:

| Feature | 0 = | 0.5 = | 1 = |
|---|---|---|---|
| `subtraction_orientation` | Entirely about adding or building | Mixed | Centrally about deleting or cutting |

After you review it, the schema is **frozen** as a Pydantic model. This fixes each feature's **name, meaning, range and position in the vector**.

**Why it must be frozen:** the comparison math matches position 1 with position 1, and so on. If two agents used different feature orders, the result would be wrong without any error. In the worked example, Run vs Cycling should have a cosine of **0.894**, but with mismatched order it comes out as **0.815**.

### 3. Sizing (code)

This stage decides how many items each agent gets, and how many agents run at once.

- **Items per agent:** k = ⌊(C × s − P) / t_max⌋, capped at `K_MAX`
- **Agents at once:** W = min(⌈N / k⌉, `W_MAX`)

| Symbol | Meaning |
|---|---|
| C | Model context window (200,000 tokens) |
| s | Safety fraction (0.5) |
| P | Prompt overhead per call (rubric + deep agent instructions) |
| t_max | Tokens for the longest item plus its output |

**Example (Musk run):** k = ⌊(200,000 × 0.5 − 6,897) / 1,459⌋ = 63, capped at 8. That gives ⌈38 / 8⌉ = **5 batches**, all 5 running at once, for about **58,000 tokens**.

### 4. Profilers (W Haiku deep agents in parallel)

Each agent scores its batch of items on the frozen schema. For every item it returns:
- a **score from 0 to 1** for each feature,
- an **evidence phrase** from the text justifying each score,
- 3–5 **tags**.

How it runs:
1. Items already in `profiles.json` are skipped, so the stage is resumable.
2. The remaining items are cut into batches of k.
3. All batches start with `asyncio.gather`, and an `asyncio.Semaphore(W)` lets at most W call the API at once.
4. Output is validated with Pydantic (`Field(ge=0, le=1)`), so an out-of-range score is rejected, not stored.
5. `profiles.json` is saved to Drive **after every batch**.

Safety nets:

| Problem | What happens |
|---|---|
| The provider rejects a request format | Structured output tries 3 ways: deep agent with `json_schema` → direct `json_schema` → plain JSON parsed by Pydantic |
| Invalid output | The batch is retried up to `MAX_RETRIES` more times |
| An item is missing from a batch | It is retried on its own at the end |
| The agent writes fields in a different order | Ignored: vectors are always built in `FEATURES` order |

**Why batches:** every call pays the prompt overhead P (about 6,900 tokens). For the Musk data, one call per item would cost about **286,000 tokens**, while batches of 8 cost about **58,000**, which is **4.9× cheaper**.

### 5. Compare (PyTorch, no LLM)

Every item becomes a vector **x** of its feature scores.

**Cosine similarity for all pairs at once:**

Xₙ = X / ‖X‖ (each row divided by its length), then **S = Xₙ Xₙᵀ**

S[a, b] is the cosine between items a and b. For N items there are N(N − 1)/2 unique pairs: 38 items give **703 pairs**, and the whole computation is one matrix multiplication.

**What two items have in common** (per-feature contribution):

contributionᵢ = (aᵢ × bᵢ) / Σⱼ (aⱼ × bⱼ)

**Worked example:**
- Run = [0.9, 0.8, 0.1, 0.1] and Football = [0.9, 0.9, 0.9, 0.1], over (physical, outdoors, social, creative)
- The products are 0.81, 0.72, 0.09 and 0.01, which sum to 1.63
- Physical contributes 0.81 / 1.63 = **50%**, and outdoors 0.72 / 1.63 = **44%**
- So the common thing is **physical, outdoor activity**

**Biggest difference:** the feature with the largest |aᵢ − bᵢ|.

**Outputs:** the top-K most similar pairs (with contributions, biggest difference and shared tags), a similarity heatmap, and a sanity check for the Musk dataset (see *Results*).

**Memory:** the full matrix needs N² × 4 bytes. N = 20,000 needs 1.6 GB, which fits on a free T4 GPU (about 15 GB). For much larger N, compute it in chunks and keep only the top-k results.

### 6. Explainer (1 Sonnet agent)

The explainer receives **only the computed numbers and evidence phrases**, and is told not to recompute or change them. For each top pair, it states the common thing and the main difference, grounded in the quoted evidence. It then summarises the patterns across all pairs. The result is saved as `report.md`.

---

## Design rules

1. **The LLM describes, code calculates.** Similarity scores never come from an LLM.
2. **Profile once, compare in math.** LLM cost grows as O(N); the O(N²) comparisons cost almost nothing.
3. **Cheap model for bulk work, strong model for judgement.** Haiku does about 90% of the calls, and Sonnet only 2.
4. **Python decides the number of agents, not the LLM**, so cost stays predictable.
5. **Every stage saves to Drive**, so a Colab disconnect never loses paid work.
6. **One dataset per run.** Each dataset gets its own schema, profiles and report, and results never mix.

---

## Setup (Google Colab)

### 1. Runtime
**Runtime → Change runtime type → T4 GPU.** Only Stage 5 uses the GPU, and it falls back to the CPU.

### 2. API key (Velona)
In Colab's 🔑 **Secrets** panel:
- add a secret named **`VELONA_API_KEY`** containing your key,
- switch **Notebook access** on.

### 3. Drive folder
```
MyDrive/similarity_pipeline/
├── items.json        ← the dataset to run (upload this)
├── schema.json       ← created by Stage 2
├── profiles.json     ← created by Stage 4
└── report.md         ← created by Stage 6
```

### 4. Run
Open the notebook and choose **Runtime → Run all**.

### Switching datasets (for example, Musk → Modi)
The notebook reads whatever is in `items.json`, so switching means swapping files:

1. Move `items.json`, `schema.json`, `profiles.json` and `report.md` into a subfolder such as `musk_results/`.
2. Upload the new dataset (for example `items_modi.json`) and **rename it to `items.json`**.
3. Choose **Run all**.

> ⚠️ **Always move the old `schema.json` and `profiles.json` out first.** Both datasets use ids starting from 1, so if the old profiles stay, the notebook treats those ids as already done and mixes the old person's scores into the new run. It would also reuse the old schema.

Skip the five-step sanity-check cell for any dataset other than Musk, because its item ids refer to the Musk file.

---

## Configuration

Set in the **Config** cell:

| Setting | Default | Purpose |
|---|---|---|
| `BASE_URL` | `https://velona.in/v1` | OpenAI-compatible endpoint (`ChatOpenAI` adds `/chat/completions`) |
| `MODEL` | `anthropic/claude-sonnet-5.5` | Schema proposal and explainer |
| `PROFILER_MODEL` | `anthropic/claude-haiku-5.5` | Stage 4 profiling |
| `SCHEMA_SAMPLE_SIZE` | 20 | Items shown to the schema agent |
| `N_FEATURES_HINT` | `'5-8'` | Number of features to propose |
| `CONTEXT_TOKENS` | 200,000 | C in the sizing formula |
| `SAFETY` | 0.5 | s in the sizing formula |
| `K_MAX` | 8 | Maximum items per agent |
| `W_MAX` | 5 | Maximum agents at once; lower it if you hit rate limits |
| `MAX_RETRIES` | 2 | Extra attempts per failed batch |
| `TOP_K_PAIRS` | 15 | Pairs shown and explained |
| `EXCLUDE_SAME_SOURCE` | `False` | `True` compares only items from different sources |

---

## Datasets

| File | Person | Items | Source |
|---|---|---|---|
| `items.json` (Musk) | Elon Musk | 38 | Lex Fridman interview excerpts, a Musk clip, a Q&A, a commentator, and a YouTube video on SpaceX's rules |
| `items_modi.json` | Narendra Modi | 65 | Lex Fridman Podcast #460 |

In the Modi items, each item contains Lex's question (marked `[Lex Fridman]`) followed by Modi's answer. Parts where only Lex speaks (the preview montage, introduction and closing monologues) are left out.

**Planned:** Jeff Bezos (Lex Fridman Podcast #405) and Steve Jobs (*Make Something Wonderful*), each run on its own.

---

## Results: Musk run (38 items)

**Schema:** Sonnet proposed 8 features. They were `action_vs_principle`, `subtraction_orientation`, `first_principles_reframing`, `sequencing_dependence`, `leadership_and_ground_involvement`, `speed_and_urgency`, `failure_mode_warning` and `technical_specificity`.

**Sanity check.** The five-step process (question requirements, delete, simplify, speed up, automate) appears in three sources, so matching steps should score as similar:
- **Mean cosine of matching steps across sources: 0.895**, compared with **0.636** for all pairs, so the pipeline recognises repeated ideas
- Best case: "delete" (item 3 vs 16) ranked **2nd of 703** pairs, with cosine 0.988
- Outlier: "speed up" item 34 ranked only 258th and 306th against its matches. That item's text is about feedback culture, so it really is different content.

**Clusters found by the explainer:** the five-step process repeated across sources, and first-principles cost reasoning (raw-material cost floor, the volume test, the platonic ideal, physics-based reasoning).

---

## Known limitations

1. **Long items get hedged scores.** Items of 700+ words that mix several ideas were scored 0.5 on almost every feature. Two such items (26 and 28) got identical vectors and a **false cosine of 1.000**. Fixes: keep items under about 500 words, and add this rule to the profiler prompt: *use 0.5 only when the 0.5 anchor fits; otherwise score near 0*.
2. **Cosine values are compressed.** All scores are 0 or above, so even unrelated items score around 0.64, and the top pairs fall between 0.94 and 1.00. Read the **rank**, not the raw value. Mean-centering each feature before the cosine would spread the values out.
3. **Tag overlap is almost always 0.** Free-form tags rarely match word for word ("deletion" vs "delete parts"). A fixed tag vocabulary defined in the schema step would fix this.
4. **The schema decides what "similar" means.** With features like `sequencing_dependence`, *different* steps of the same process can score as high as the *same* step in different sources, because they share structure. Choose features to match the question you're asking.
5. **Velona + Sonnet rejects forced tool calls.** That's why structured output uses `json_schema`, not tool calling. Haiku supports both.

---

## Dependencies

```
deepagents          # LangChain Deep Agents (create_deep_agent)
langchain-openai    # ChatOpenAI for the OpenAI-compatible Velona endpoint
pydantic            # schema and output validation
torch               # Stage 5 comparisons (preinstalled in Colab)
matplotlib          # heatmap (preinstalled in Colab)
pdfplumber          # PDF extraction (Stage 1, done outside the notebook)
```

The `google-auth` version warning that pip prints in Colab is harmless.
