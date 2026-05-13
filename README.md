# ELF: Embedded Language Flows

[![arXiv](https://img.shields.io/badge/arXiv-2605.10938-b31b1b.svg)](https://arxiv.org/abs/2605.10938)&nbsp;
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)&nbsp;
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-ELF-blue.svg)](https://huggingface.co/embedded-language-flows)&nbsp;
[![Github Official](https://img.shields.io/badge/github-JAX-official.svg)](https://github.com/lillian039/ELF)&nbsp;

This is the **Unofficial PyTorch port** for the paper *ELF: Embedded Language Flows*. This code runs on GPU.

ELF is a class of continuous diffusion language models based on continuous-time Flow Matching. Unlike existing DLMs, ELF predominantly stays within the continuous embedding space until the final time step, where it maps to discrete tokens using a shared-weight network. This formulation makes it straightforward to adapt established techniques from image-domain diffusion models, e.g., classifier-free guidance (CFG).

<p align="center">
  <img src="assets/teaser.gif" alt="Conceptual illustration of ELF" width="100%"/>
</p>
<p align="left">
  <em><strong>Conceptual illustration of ELF.</strong> Orange points denote data represented in continuous embedding space, and purple lines show denoising trajectories from Gaussian noise to clean embeddings. Discretization is applied only at the final time step (t=1) using a shared-weight network.</em>
</p>

<p align="center">
  <img src="assets/generation.gif" alt="Denoising trajectory of ELF-B" width="100%"/>
</p>
<p align="left">
  <em><strong>Denoising trajectory</strong> of ELF-B. As t increases from 0 to 1, ungrammatical sentences are progressively refined into fluent and grammatical text.</em>
</p>

<p align="center">
  <img src="assets/sys_compare.jpg" alt="System-level comparison" width="100%"/>
</p>
<p align="left">
  <em><strong>System-level comparison.</strong> ELF-B outperforms both discrete and continuous DLMs trained under similar settings (a) and distilled variants of other baselines that require additional rounds of training (b), and uses substantially fewer training tokens (c).</em>
</p>

## Initialization

Install the dependencies (PyTorch+GPUs) and log in to WandB to track your experiments if needed.

```bash
pip install -r requirements.txt
wandb login YOUR_WANDB_API_KEY
```

## Inference

You can quickly verify your setup with converted pytorch checkpoints from JAX.

[ELF-B-xsum/elf_model.pt](https://huggingface.co/Xrenya/ELF-B-xsum)  
[ELF-B-owt/elf_model.pt](https://huggingface.co/Xrenya/ELF-B-owt/)  
[t5_small_encoder/t5_encoder.pt](https://huggingface.co/Xrenya/t5_small_encoder)  

```bash
python scripts/generate.py \
  --config <PATH>/ELF-B-owt/ELF-B-owt.yml \
  --elf-checkpoint <PATH>/ELF-B-owt/elf_model.pt \
  --batch-size 1 \
  --method ode \
  --steps 32 \
  --cfg 1 \
  --self-cond-cfg 3 \
  --sde-gamma 1.5
# sampling: {'method': 'ode', 'steps': [32], 'cfgs': [1.0], 'self_cond_cfg_scales': [3.0], 'time_schedule': 'logit_normal', 'sde_gamma': 1.5}
# {"id": 0, "generated": "The Redskins continued to move relatively aggressively—midway to the game when a deep cross by Jacques Andrivsson risen off the net that prevented Sacramento's right-hander Marc Menier to move the ball off the net. (+ 6) The difficulty for the team decreased in the fifth-minute, as a far receiving tackle by Jeymek Ezmanpeas fell off the net and forced Boston back into the middle of the movement. Without the ability to balance the two tackles, Boston's forwards were offered an unusual opportunity to move the ball off the net and the Redskins stopped their movement aggressively too. Boston maintained that strategy with Lewis's first reposte goal in Saturday's 21–2 win over Culi Rothomo. The Redskins managed to establish themselves relatively easily, scoring a deep cross by Jeremy Lewis to create what was Boston's best goal of the season. Burques Weddleland scored the second goal and Stanford's Allahas Cal Vicgot returned for sixth. (+ 6) Despite the attack that saved San Francisco's 2-0, things finally went well for the Redskins, as their goalkeepers Akal Abshan and Jason Acosta defied to their defenses afterecuring the first spot in the MLS series, where they are second tied this season. The game should have bailed off against Madrid, who opened up a rogue 3-0 at Peterron Park. The Redskins would have had the kind of vulnerability of an unfortunate oppositioning team, and 21 movements -- including a cross and a throw -- were postponed this season. The Redskins kept their heads tight before anything even rummed to happen. In the second-half, opposition from a large crowd emerged, but Alex Cloud's rapid crossing goal gave Boston a minor advantage in the spotless-point win in St. City. Unappelled Play The game was marked by two rushes: one outside of the midfield and another outside of the midfield. Boston was unable to protect center Darlin Bergstrandron after hitting the late to rush for the right-hander, while left-back Jeney Hill crunced a header off Sacramento's crossing. Boston did not deserve much of the same, but Hill's size and a flighty farce half-time ensure that Boston had a protecting player's advantage. Media playback is not supported on this device The Detroit Azers' one terrible performance in their 2-0 win over Stanford's Opta was the first Sacramento's won the season. Chris Harrison has not played again for Boston. El State's Chris Harrison is arguably suffering from his deceptive performance but also struggling after Chris Harrison' assists in Saturday's win. Madrid's struggle in Unappeared Comparisons Madrid's struggle was not playable, especially after falling with ground during the losing season. Former Madrid forward Chris Harrison suffered an inappropriate cross tackle to earn his courageous first reposte goal in Madrid's 3-2 win over Culi Rothomo. :-) Media playback is not supported on this device 0:14 Sacramento, above a penalty kick with just 15 seconds left before Boston's Dalton Lewis collected a long ball as his team plumbed to a defying finish (Z Center/span) Although the rehabilitated Redskins have since gone scoreless in their first games of the season, there is little doubt over what Boston's woes remain. Even though it was their first playoff loss in a season since July 2001, it has not been their first of the season. Boston has struggled in the losing of four games, as Madrid has been desperate, yet Alex Cloud's first substitute goal in Boston's 2-0 win over Guantmo State was such a negative effect that Cole left the team on a** his previous injury issues with Madrid. Media playback is not supported on this device Boston's club's response to the Redskins' spotless performance in Saturday's win over St. City by MLS's Steven Culver due to his assessment of the game blam any part of this situation. The club's deceptive performance has had a proportionate advantage between the Redskins and St. City over their last four games in Madrid. By that account, it has been the last one-game loss between these two ill-popular clubs. Media playback is not supported on this device Captain Robinson compares the struggle between Boston's Detroit Azers and the city's Jeymek-Ezmanpeas, who left things nearly without compassion in their 2-0 win over St. City"}


python scripts/generate.py \
  --config <PATH>/ELF-B-xsum/ELF-B-xsum.yml \
  --elf-checkpoint <PATH>/ELF-B-xsum/elf_model.pt \
  --encoder-checkpoint t5_small_encoder/t5_encoder.pt \
  --prompt "The UK government has announced a new package of measures aimed at reducing household energy bills. Ministers said the plan would expand subsidies for low-income families and invest in insulation for older homes. Opposition parties welcomed parts of the proposal but said it did not go far enough to address rising costs. Energy companies are expected to meet officials next week to discuss how the scheme will be delivered." \
  --batch-size 1 \
  --method ode \
  --steps 64 \
  --cfg 1 \
  --self-cond-cfg 1

# sampling: {'method': 'ode', 'steps': [64], 'cfgs': [1.0], 'self_cond_cfg_scales': [1.0], 'time_schedule': 'logit_normal', 'sde_gamma': 0.0}
# {"id": 0, "generated": "Plans to build a range of energy insulation for families and families have been proposed to the UK government."}

```


<table><tbody>
<td valign="bottom">OpenWebText (unconditional)</td>
<td valign="bottom" align="center">ELF-B (105M)</td>
<td valign="bottom" align="center">ELF-M (342M)</td>
<td valign="bottom" align="center">ELF-L (652M)</td>
<tr><td align="left">pre-trained checkpoint</td>
<td align="center"><a href="https://huggingface.co/Xrenya/ELF-B-owt">ELF-B-owt</a></td>
<td align="center"><a href="https://huggingface.co/embedded-language-flows/ELF-M-owt">ELF-M-owt</a></td>
<td align="center"><a href="https://huggingface.co/embedded-language-flows/ELF-L-owt">ELF-L-owt</a></td>
</tr>
<tr><td align="left">Sampling steps (SDE)</td>
<td align="center">32</td>
<td align="center">64</td>
<td align="center">64</td>
</tr>
<tr><td align="left">Gen. PPL ↓ (paper)</td>
<td align="center">24.1</td>
<td align="center">21.7</td>
<td align="center">23.3</td>
</tr>
<tr><td align="left">Entropy ↑ (paper)</td>
<td align="center">5.15</td>
<td align="center">5.18</td>
<td align="center">5.28</td>
</tr>
</tbody></table>

<table><tbody>
<td valign="bottom">Conditional generation (ELF-B)</td>
<td valign="bottom" align="center">WMT14 De-En</td>
<td valign="bottom" align="center" colspan="3">XSum</td>
<tr><td align="left">pre-trained checkpoint</td>
<td align="center"><a href="https://huggingface.co/embedded-language-flows/ELF-B-de-en">ELF-B-de-en</a></td>
<td align="center" colspan="3"><a href="https://huggingface.co/Xrenya/ELF-B-xsum">ELF-B-xsum</a></td>
</tr>
<tr><td align="left">Metric</td>
<td align="center">BLEU ↑</td>
<td align="center">ROUGE-1 ↑</td>
<td align="center">ROUGE-2 ↑</td>
<td align="center">ROUGE-L ↑</td>
</tr>
<tr><td align="left">Score (paper)</td>
<td align="center">26.4</td>
<td align="center">36.0</td>
<td align="center">12.2</td>
<td align="center">27.8</td>
</tr>
</tbody></table>

Slight differences in metrics may arise from different compute setups. Our results were computed on TPU v5p-64.

#### Sanity Check

1. **Get the checkpoint.** All pre-trained checkpoints are on HuggingFace under [`embedded-language-flows`](https://huggingface.co/embedded-language-flows) and are pulled automatically via `--checkpoint_path <hf-repo-id>` — no manual download needed. To use a locally trained checkpoint, pass the path to the specific checkpoint file, e.g. `--checkpoint_path outputs/elf_b-owt/checkpoint_19000`.

2. **(Optional) Tweak the config.** The provided `configs/training_configs/train_owt_ELF-{B,M,L}.yml` already point at the correct HuggingFace data + T5 encoder, so they run as-is. You may want to edit:
    - `output_dir` — where samples and logs are written
    - `wandb_entity` — set to your entity, or set `use_wandb: false` to disable
    - `sampling_configs_path` — defaults to `configs/sampling_configs/uncond_sampling_configs.yml` (32-step SDE + 64-step SDE, both with self-conditioning CFG); swap for your preferred schedule if needed

3. **Launch evaluation.**

**Unconditional generation:**
```bash
cd src/

  # ELF-B (105M)
  python eval.py \
      --config configs/training_configs/train_owt_ELF-B.yml \
      --checkpoint_path embedded-language-flows/ELF-B-owt

  # ELF-M (342M) — smaller batch to fit the bigger model
  python eval.py \
      --config configs/training_configs/train_owt_ELF-M.yml \
      --checkpoint_path embedded-language-flows/ELF-M-owt \
      --config_override global_batch_size=64

  # ELF-L (652M)
  python eval.py \
      --config configs/training_configs/train_owt_ELF-L.yml \
      --checkpoint_path embedded-language-flows/ELF-L-owt \
      --config_override global_batch_size=64
```
The evaluator generates 1,000 samples and reports Gen. PPL (under a pretrained GPT-2 Large) and unigram entropy. Expected: Gen. PPL ≈ 24 and entropy ≈ 5.15 for ELF-B at 32 SDE steps.

**Conditional generation:**
```bash
cd src/

# XSum (summarization)
python eval.py \
    --config configs/training_configs/train_xsum_ELF-B.yml \
    --checkpoint_path embedded-language-flows/ELF-B-xsum

# WMT14 De-En (translation)
python eval.py \
    --config configs/training_configs/train_de-en_ELF-B.yml \
    --checkpoint_path embedded-language-flows/ELF-B-de-en
```
The evaluator runs on each task's **validation** set and reports BLEU for WMT14 De-En and ROUGE-1/2/L for XSum. Expected: BLEU ≈ 26.7 on De-En; ROUGE-1/2/L ≈ 36.3 / 12.5 / 28.1 on XSum. Note that the paper numbers are computed on the **test** sets, so validation scores here may differ slightly.

## Data Preparation

Three task settings: unconditional generation on **OpenWebText**, machine translation on **WMT14 De-En**, and summarization on **XSum**. All use a frozen T5 encoder for text-to-embedding mapping.

#### Pre-tokenized splits

We provide pre-tokenized splits (T5 tokenizer) and the JAX T5-small encoder on HuggingFace under [`embedded-language-flows`](https://huggingface.co/embedded-language-flows). They are loaded directly via `datasets.load_dataset` — no manual download needed. Defaults wired into the configs:

| Task | `data_path` / `eval_data_path` |
| --- | --- |
| OpenWebText | `embedded-language-flows/openwebtext-t5` |
| WMT14 De-En | `embedded-language-flows/wmt14_de-en_{train,validation}_t5` |
| XSum | `embedded-language-flows/xsum_{train,validation}_t5` |
| T5 encoder | `Xrenya/t5_small_encoder/t5_encoder.pt` |

To use a local copy, point `data_path` at a directory saved with `datasets.save_to_disk` — the loader falls back to `load_from_disk`.

#### Prepare your own data

To train on a custom dataset, pre-tokenize it with the tokenizer and save it as a HuggingFace `Dataset` (Arrow).

**Unconditional generation** (e.g., OWT): each example needs only `input_ids` — the token ids of the text to be generated.

**Conditional generation** (e.g., translation, summarization): each example needs both `input_ids` (target/output text) and `condition_input_ids` (source/input text, e.g., the German sentence or the article). The collator prepends `condition_input_ids` to `input_ids` and builds the appropriate attention masks automatically.

Minimal recipe:

```python
from datasets import Dataset
from transformers import T5Tokenizer

tok = T5Tokenizer.from_pretrained("google-t5/t5-small")

# Unconditional
def encode_uncond(ex):
    return {"input_ids": tok(ex["text"], add_special_tokens=False)["input_ids"]}

# Conditional (translation / summarization)
def encode_cond(ex):
    return {
        "condition_input_ids": tok(ex["source"], add_special_tokens=False)["input_ids"],
        "input_ids": tok(ex["target"], add_special_tokens=False)["input_ids"],
    }

ds = Dataset.from_list(my_examples).map(encode_uncond, remove_columns=...)  # or encode_cond
ds.save_to_disk("/path/to/my_dataset")
```

Then point your config at it:

```yaml
data_path: /path/to/my_dataset
eval_data_path: /path/to/my_eval_dataset   # optional
```

For evaluation-only JSONL inputs (raw text, tokenized at load time), see `load_jsonl_dataset` in [data_utils.py:110-130](src/utils/data_utils.py#L110-L130) — set `eval_data_path` to a `.jsonl` file with one `{"input": ..., "output": ...}` example per line.

## Training

Run the following command to launch training:

```bash
python train.py --config configs/training_configs/train_owt_ELF-B.yml
```

Available training configs:

- `configs/training_configs/train_owt_ELF-B.yml` — unconditional generation on OpenWebText, ELF-B (default)
- `configs/training_configs/train_owt_ELF-M.yml` — unconditional generation on OpenWebText, ELF-M
- `configs/training_configs/train_owt_ELF-L.yml` — unconditional generation on OpenWebText, ELF-L
- `configs/training_configs/train_de-en_ELF-B.yml` — WMT14 De-En machine translation
- `configs/training_configs/train_xsum_ELF-B.yml` — XSum abstractive summarization

Default ELF-B training uses Muon at blr=0.001 (base learning rate; effective lr = blr × batch_size / 256 = 0.002 at the default batch size of 512), global batch size 512, and runs 5 epochs on OWT (~95K steps) on TPU v5p-64 (~1.5 h per epoch).

#### Config System

The training system uses two config layers:

- **`configs/config.py`** — base `Config` dataclass with all default hyperparameters
- **`configs/training_configs/*.yml`** — task-specific overrides loaded by `load_config_from_yaml()`

The system merges these, allowing you to customize only the parameters you need.

#### Customizing Training

To create a custom experiment:

1. **Create a new config file** (e.g., `configs/training_configs/my_exp.yml`)
2. **Launch with your config:**
   ```bash
   python train.py --config configs/training_configs/my_exp.yml
   ```

**Example custom config:**

```yaml
model: ELF-M                # Use ELF-M model (342M)

epochs: 4
global_batch_size: 512
blr: 0.002
optimizer: muon

denoiser_p_mean: -1.5       # Logit-normal time schedule
denoiser_p_std: 0.8
denoiser_noise_scale: 2.0
self_cond_prob: 0.5
decoder_prob: 0.2           # 20% decoding (CE) / 80% denoising (L2)
```

For more details on configuration options, refer to `config.py` and the YAML files under `configs/training_configs/`.

#### Sampling Configuration

Sampling is decoupled from training and is controlled by a separate YAML in `configs/sampling_configs/`, referenced from each training config via `sampling_configs_path`:

- `uncond_sampling_configs.yml` — unconditional generation: two SDE schedules, 32-step (γ=1.5) and 64-step (γ=1.0), both with SC-CFG=3
- `cond_sampling_configs.yml` — conditional generation (translation / summarization): one 64-step ODE schedule with CFG=2 and SC-CFG=1

Each list entry specifies a sampler (`ode` / `sde`), `num_sampling_steps`, `cfgs`, `self_cond_cfg_scales`, and `time_schedule`. The evaluator iterates through all entries.

## Checkpointing

Checkpoints are saved at the end of each epoch (or at fractional intervals if `save_freq < 1`) to `output_dir/checkpoint_<step>`, keeping up to 10 recent checkpoints. Only process 0 writes to disk.

If `hf_repo_id` is set in the config, the entire `output_dir` is uploaded to HuggingFace after each save.

**Auto-resume:** if `--resume` is not specified, training automatically detects and resumes from the latest checkpoint in `output_dir`.

**Loading:** `load_checkpoint` accepts a local path or an HF repo ID (e.g., `embedded-language-flows/ELF-B-owt`). For a directory, it uses the latest checkpoint inside.

The T5 encoder weights (`encoder_checkpoint`) are stored separately as a `.pt` file and loaded once at startup. They can also be specified as an HF path (default: `Xrenya/t5_small_encoder/t5_encoder.pt`).

## License

This repo is under the MIT license. See [LICENSE](LICENSE) for details.

## Citation

If you find this work useful in your research, please consider citing our paper :)

```bib
@article{elf2026,
  title={ELF: Embedded Language Flows},
  author={Hu, Keya and Qiu, Linlu and Lu, Yiyang and Zhao, Hanhong and Li, Tianhong and Kim, Yoon and Andreas, Jacob and He, Kaiming},
  journal={arXiv preprint arXiv:2605.10938},
  year={2026}
}
```

## Acknowledgement

We gratefully acknowledge the Google TPU Research Cloud (TRC) for granting TPU access.
We hope this work will serve as a useful resource for the open-source community.
