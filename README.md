# GKANSU_Improved — FastKAN Pruning Notebooks

This README walks through what the code in each notebook actually does,
block by block, with the real functions pulled from the notebooks and a
plain-English explanation under each one. It's meant to be read alongside
the `.ipynb` files, not instead of them.

All four notebooks prune the same two layers — `fast_kan1` and `fast_kan2` —
inside `GKANSU_Improved`. Nothing else in the model (`input_proj`, the GAT
layers, `ssa`, `final_refinement`) is ever touched by any of the four
methods.

```
Notebook              Method    Scores each channel by...
─────────────────────────────────────────────────────────────────
Taylor_prune.ipynb    Taylor    (gradient × activation)²  [Molchanov et al.]
wsap.ipynb             WSAP     (‖w_base‖₁+‖w_spline‖₁) × σ(activation)
shap__prune.ipynb     ShapKAN   Shapley value via permutation sampling
kanko_prune.ipynb     KAN-KO    loss increase when the channel is zeroed
```

---

## 0. Shared setup every notebook depends on

```python
train_dataset = HSI_GraphDataset(trainSL, trainSH)
test_dataset  = HSI_GraphDataset(testSL, testSH)
train_loader  = GeometricDataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader   = GeometricDataLoader(test_dataset,  batch_size=8, shuffle=True)
```
`train_loader` shuffling is fine — it's only used to fine-tune. `test_loader`
shuffling caused a real bug in KAN-KO (see §5) — **set `shuffle=False` on
`test_loader`** before trusting any batch-mean-style metric across runs.

Every notebook loads the same trained checkpoint before scoring:
```python
CHECKPOINT = '/content/drive/MyDrive/GKANSU/model_checkpoint.pth'
model = GKANSU_Improved(in_channels=20, out_channels=200, img_size=(64, 64)).to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
```
This has to happen **fresh, right before scoring/evaluating**, not just once
somewhere earlier in the session — see §5 for why ShapKAN originally missed
this and what it broke.

Fine-tuning everywhere uses a combined loss, not SAM alone:
```python
def sam_loss_fn(hr, sr, eps=1e-8):
    hr_flat = hr.view(*hr.shape[:2], -1)
    sr_flat = sr.view(*sr.shape[:2], -1)
    cos = (hr_flat * sr_flat).sum(1, keepdim=True) / (
        hr_flat.norm(dim=1, keepdim=True) * sr_flat.norm(dim=1, keepdim=True) + eps)
    return torch.acos(cos.clamp(-1, 1)).mean()

def loss_fn(sr, hr, alpha=0.7):
    return alpha * sam_loss_fn(hr, sr) + (1 - alpha) * F.mse_loss(sr, hr)
```
`alpha=0.7` weights spectral angle more heavily than pixel-wise MSE. This
70/30 mix replaced an earlier SAM-only loss that caused a real training
failure (§5, WSAP).

---

## 1. Taylor (`Taylor_prune.ipynb`)

**Idea:** a channel's importance is approximated by how much the loss would
change if you removed it, using a first-order Taylor expansion:
`I(w) ≈ (gradient × weight)²`. Cheap to compute — just needs one backward
pass, no repeated forward passes per channel.

### Scoring — hooking the backward pass

```python
class TaylorPrunerKAN:
    def __init__(self, model, prune_ratio=0.30):
        self.model, self.prune_ratio = model, prune_ratio
        self.hooks, self.masks = [], {}
        self._saved_acts = {}
        self.scores  = defaultdict(float)
        self.counts  = defaultdict(int)

    def _register_hooks(self):
        def fwd1(mod, inp, out):
            self._saved_acts['fast_kan1'] = out          # cache activation
        def bwd1(mod, grad_in, grad_out):
            act, grad = self._saved_acts.get('fast_kan1'), grad_out[0]
            if act is not None and grad is not None:
                # per-channel Taylor score = mean(|grad * activation|) over H,W and batch
                per_sample = (grad * act).mean(dim=2).mean(dim=2).abs()
                score = per_sample.mean(dim=0)
                self.scores['fast_kan1'] += score.detach().cpu()
                self.counts['fast_kan1'] += 1
        # (fast_kan2 mirrors fast_kan1)
        self.hooks.append(self.model.fast_kan1.register_forward_hook(fwd1))
        self.hooks.append(self.model.fast_kan1.register_full_backward_hook(bwd1))
```
`register_forward_hook` grabs the layer's **output activation** on the way
forward; `register_full_backward_hook` grabs the **gradient** flowing back
into that same output on the way backward. Multiplying them together and
averaging over the spatial dims (H, W) and the batch gives one score per
channel — this is the `(gradient × weight)` product from the Taylor
formula, computed on activations rather than raw weights (a standard
variant, since it captures how much that channel's *output* actually
mattered for this batch, not just its static weight magnitude).

```python
    def collect(self, train_loader, device, n_batches=30):
        self._register_hooks()
        self.model.train()
        for i, (graph_data, hr_imgs) in enumerate(train_loader):
            if i >= n_batches: break
            self.model.zero_grad()
            sr = self.model(graph_data.x, graph_data.edge_index)
            loss = loss_fn(sr.view(hr_imgs.shape), hr_imgs)
            loss.backward()          # <- hooks fire here, accumulating scores
            # NOTE: optimizer.step() must NOT be called in this loop —
            # see §5, this was a real bug in an earlier version.
        self._remove_hooks()
```
30 mini-batches are run purely to accumulate gradient statistics — the model
is *not* being trained here, just probed. `loss.backward()` triggers every
registered hook, which is where `self.scores` actually gets filled in.

```python
    def compute_masks(self):
        for name in ['fast_kan1', 'fast_kan2']:
            avg_score = self.scores[name] / self.counts[name]
            n_prune = max(1, int(avg_score.shape[0] * self.prune_ratio))
            prune_idx = torch.argsort(avg_score)[:n_prune]   # lowest scores first
            keep_mask = torch.ones(avg_score.shape[0], dtype=torch.bool)
            keep_mask[prune_idx] = False
            self.masks[name] = keep_mask
```
Average the accumulated score over however many batches it was collected
from, sort ascending, and mark the bottom `prune_ratio` fraction as "not
kept." Standard lowest-score-gets-cut logic — the same pattern reused with
different scores in the other three notebooks.

```python
    def apply_masks(self):
        for name in ['fast_kan1', 'fast_kan2']:
            mask, module = self.masks[name], getattr(self.model, name)
            def make_hook(m):
                def hook(mod, inp, out):
                    return out * m.float().to(out.device).view(1, -1, 1, 1)
                return hook
            self.hooks.append(module.register_forward_hook(make_hook(mask)))
```
This is the **soft mask**: a forward hook that multiplies the layer's output
by a 0/1 mask, broadcast across the spatial dimensions. The tensor shape
never changes — pruned channels just output zero from here on. This is why
Taylor (like ShapKAN and KAN-KO) reports *sparsity*, not a smaller model:
nothing about the layer's actual size changed, only its output values.

### The sweep

```python
prune_ratios = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
for ratio in prune_ratios:
    pruner = TaylorPrunerKAN(model, prune_ratio=ratio)
    model.load_state_dict(torch.load(CHECKPOINT))   # reload clean weights every ratio
    model.to(device)
    pruner.collect(train_loader, device, n_batches=30)
    pruner.compute_masks()
    base_m = evaluate_full_metrics(model, test_loader, device)   # before mask
    pruner.apply_masks()
    prune_m = evaluate_full_metrics(model, test_loader, device)  # after mask, no FT
    # ... 10 epochs of fine-tuning with loss_fn ...
    ft_m = evaluate_full_metrics(model, test_loader, device)     # after fine-tune
    pruner.remove_masks()
```
The checkpoint reload at the top of every iteration is what keeps ratios
independent — without it, the 40% run would start from whatever the 30%
run's fine-tuning left behind, not the original trained model.

---

## 2. WSAP (`wsap.ipynb`)

**Idea:** decide *how much* to prune a layer from its weight statistics
(Coefficient of Variation), score channels using both weight paths plus
real activation spread, and — unlike the other three methods — actually
**rebuild a physically smaller model**.

### Per-layer sparsity ratio from CoV

```python
CoV = std(all weights in layer) / abs(mean(all weights in layer))
if CoV > τ_high:  ratio = 0.7   # high dispersion → assume redundant → prune hard
elif CoV > τ_low: ratio = 0.5
else:             ratio = 0.3  # tight, consistent weights → prune conservatively
```
A layer whose weights are large-but-inconsistent (high CoV) is treated as
more redundant than one with small, consistent weights.

### Channel score

```python
def compute_wsap_scores(kan_layer, act_list):
    w_base_l1   = kan_layer.base_conv[0].weight.data.abs().mean(dim=(1,2,3))
    w_spline_l1 = kan_layer.spline_conv[0].weight.data.abs().mean(dim=(1,2,3))
    weight_score = w_base_l1 + w_spline_l1
    all_acts = torch.cat(act_list, dim=0)
    act_std  = all_acts.std(dim=(0, 2, 3)).to(weight_score.device)
    return weight_score * act_std, weight_score, act_std
```
`act_list` comes from forward hooks recording real activations over 20
calibration batches beforehand. A channel with both large average weights
*and* high activation variance across real data is scored as important;
small-weight, near-constant-output channels score low and get pruned.

### The part that makes WSAP different: physically rebuilding the model

```python
class PrunedGKANSU(nn.Module):
    def __init__(self, orig_model, keep_idx1, keep_idx2):
        super().__init__()
        n1, n2 = len(keep_idx1), len(keep_idx2)
        hidden_dim = orig_model.fast_kan1.base_conv[0].weight.shape[1]

        # everything except the two FastKAN blocks is copied AS-IS
        self.input_proj       = copy.deepcopy(orig_model.input_proj)
        self.gat_layers       = copy.deepcopy(orig_model.gat_layers)
        self.batch_norms      = copy.deepcopy(orig_model.batch_norms)
        self.ssa               = copy.deepcopy(orig_model.ssa)
        self.final_refinement = copy.deepcopy(orig_model.final_refinement)

        # fast_kan1: slice out only the rows for kept output channels
        self.fast_kan1 = copy.deepcopy(orig_model.fast_kan1)
        self.fast_kan1.base_conv[0].weight.data = \
            orig_model.fast_kan1.base_conv[0].weight.data[keep_idx1]
        self.fast_kan1.spline_conv[0].weight.data = \
            orig_model.fast_kan1.spline_conv[0].weight.data[keep_idx1]

        # fast_kan2: slice BOTH its output rows (keep_idx2) AND its input
        # columns (keep_idx1), since its input is fast_kan1's shrunk output
        G = orig_model.fast_kan2.rbf.grid.shape[0]
        spline_in_idx = torch.cat([keep_idx1 * G + g for g in range(G)]).sort().values
        self.fast_kan2 = copy.deepcopy(orig_model.fast_kan2)
        self.fast_kan2.base_conv[0].weight.data = \
            orig_model.fast_kan2.base_conv[0].weight.data[keep_idx2][:, keep_idx1]
        self.fast_kan2.spline_conv[0].weight.data = \
            orig_model.fast_kan2.spline_conv[0].weight.data[keep_idx2][:, spline_in_idx]

        # bridge the channel-count mismatch this creates at the residual adds
        self.res_proj1     = nn.Conv2d(hidden_dim, n1, kernel_size=1, bias=False)
        self.res_proj2     = nn.Conv2d(n1, n2,        kernel_size=1, bias=False)
        self.restore_proj  = nn.Conv2d(n2, hidden_dim, kernel_size=1, bias=False)
```
This is the key structural move. `fast_kan1` and `fast_kan2` both have a
residual connection (`output = ReLU(fast_kan(x) + x)`), which normally
requires the output to have the exact same channel count as the input. If
you just slice out rows to shrink the layer, that addition breaks — shapes
no longer match. WSAP's fix is three small 1×1 convolutions that project the
residual (and the final output) to whatever size is needed at each step, so
the addition still works with a genuinely smaller `fast_kan1`/`fast_kan2` in
the middle. `ssa` and `final_refinement` still expect the original
`hidden_dim` (128), so `restore_proj` expands back to 128 right before them.

```python
    def forward(self, x, edge_index):
        x = self.input_proj(x)
        for gat_layer, bn in zip(self.gat_layers, self.batch_norms):
            residual = x
            x = F.relu(bn(gat_layer(x, edge_index)) + residual)
        x = x.view(batch_size, H, W, -1).permute(0, 3, 1, 2)

        res1 = self.res_proj1(x)          # project 128 → n1 to match fast_kan1's new output
        x    = F.relu(self.fast_kan1(x) + res1)

        res2 = self.res_proj2(x)          # project n1 → n2 to match fast_kan2's new output
        x    = F.relu(self.fast_kan2(x) + res2)

        x = F.relu(self.restore_proj(x))  # n2 → 128, back to what ssa/final_refinement expect
        x = self.ssa(x)
        x = self.final_refinement(x)
        return torch.clamp(x, 0, 1)
```
Result: **5,013,457 → 3,375,057 parameters (−32.7%)**, from shrinking only
`fast_kan1`/`fast_kan2` and adding three small projection convs — nothing
else in the network changes shape.

---

## 3. ShapKAN (`shap__prune.ipynb`)

**Idea:** treat the 128 channels of a layer as players in a cooperative game
and compute each one's Shapley value — its average marginal contribution
across many random orderings of the other channels.

### The value function — what "contribution" means here

```python
def value_function_fastkan(model, layer_name, channel_mask, calib_graph, calib_hr, device):
    mask_t = torch.tensor(channel_mask, dtype=torch.float32, device=device)
    def hook_fn(module, inp, out):
        return out * mask_t.view(1, -1, 1, 1)     # zero every channel NOT in the coalition
    handle = getattr(model, layer_name).register_forward_hook(hook_fn)
    with torch.no_grad():
        sr_img = model(calib_graph.x, calib_graph.edge_index).view(calib_hr.shape)
        # SAM between this masked output and ground truth IS the coalition's value
        dot = (calib_hr.view(*calib_hr.shape[:2],-1) * sr_img.view(*sr_img.shape[:2],-1)).sum(1, keepdim=True)
        ...
        sam_val = torch.acos(cos).mean().item()
    handle.remove()
    return sam_val
```
**This is the single most important line to understand about this
notebook:** `v(S)` — the "value" of a coalition of channels — is computed as
**SAM specifically**, not the combined SAM+MSE loss used everywhere else.
That means every Shapley value this notebook produces measures "how much
does this channel matter for spectral angle," and says nothing directly
about pixel-level (RMSE) fidelity. This is why the corrected results (§5)
show SAM staying flat while RMSE gets consistently worse — the scoring
function was never asked about RMSE in the first place.

### Approximating Shapley values (exact computation is exponential)

```python
def shapley_value_approx_fastkan(model, layer_name, n_channels, calib_graph, calib_hr,
                                  device, n_permutations=32, antithetical=True, seed=42, batch_size=4):
    for it in range(n_loops):
        permutations = _gen_permutations(bsz, d, rng, antithetical)  # random channel orderings
        S = np.zeros((bsz, d), dtype=bool)          # coalition-so-far, per sampled permutation
        prev_values = np.zeros(bsz)                  # v(∅) = 0
        for pos in range(d):                          # walk each permutation left to right
            S[:, permutations[:, pos]] = True
            next_values = [value_function_fastkan(..., mask) for mask in S]  # v(S ∪ {channel})
            deltas[:, permutations[:, pos]] = next_values - prev_values      # marginal contribution
            prev_values = next_values
        # Welford running mean update — accumulates avg marginal contribution per channel
```
Exact Shapley values need `2^128` coalition evaluations per channel —
completely infeasible. Instead, 32 random permutations of the 128 channels
are sampled; for each one, channels are added one at a time in that order,
and each channel's marginal contribution (`v(S∪{i}) − v(S)`) at the position
it appears is recorded. Averaging these marginal contributions over enough
random orderings converges to the true Shapley value. `antithetical=True`
pairs each random permutation with its reverse, which reduces variance for
the same sampling budget (a standard variance-reduction trick, not unique to
this implementation).

### Selecting and applying the mask

```python
def keep_mask_from_sv(shapley_vals, prune_ratio, n_channels):
    abs_sv = np.abs(shapley_vals)
    prune_idx = set(np.argsort(abs_sv)[:int(np.ceil(n_channels * prune_ratio))])
    return np.array([i not in prune_idx for i in range(n_channels)])

def apply_permanent_hook(model, layer_name, keep_mask):
    mask_t = torch.tensor(keep_mask, dtype=torch.float32)
    def hook(module, inp, out):
        return out * mask_t.to(out.device).view(1, -1, 1, 1)
    return getattr(model, layer_name).register_forward_hook(hook)
```
Same soft-mask pattern as Taylor and KAN-KO — lowest `|Shapley value|`
channels get zeroed via a forward hook, tensor shape unchanged.

---

## 4. KAN-KO (`kanko_prune.ipynb`)

**Idea:** directly measure how much the loss goes up when a channel is
knocked out — no approximation, no gradient, just brute-force ablation, one
channel at a time.

```python
def compute_kanko_scores(model, data_loader, device, n_batches=30):
    ko_scores = {'fast_kan1': torch.zeros(n_ch1), 'fast_kan2': torch.zeros(n_ch2)}

    def run_loss(graph_data, hr_imgs, hook_fn=None):
        hooks = [getattr(model, ln).register_forward_hook(fn) for ln, fn in (hook_fn or {}).items()]
        with torch.no_grad():
            sr = model(graph_data.x, graph_data.edge_index)
            loss = loss_fn_ko(sr.view(hr_imgs.shape), hr_imgs).item()
        for h in hooks: h.remove()
        return loss

    for layer_name, n_ch in [('fast_kan1', n_ch1), ('fast_kan2', n_ch2)]:
        for k in range(n_ch):
            ko_sum = 0.0
            for gd, hr in batch_data:
                baseline = run_loss(gd, hr)                       # loss with NOTHING removed
                def make_ko_hook(ch_idx):
                    def hook(mod, inp, out):
                        out = out.clone(); out[:, ch_idx, :, :] = 0.0   # zero exactly one channel
                        return out
                    return hook
                ko_loss = run_loss(gd, hr, hook_fn={layer_name: make_ko_hook(k)})
                ko_sum += (ko_loss - baseline)      # how much worse did the loss get?
            ko_scores[layer_name][k] = ko_sum / len(batch_data)
```
This is literally `(n_ch1 + n_ch2 + 1) × n_batches` forward passes — for
each of the 256 total channels across both layers, run the full model with
just that channel zeroed and compare the loss to the unmodified baseline. A
channel whose removal barely changes the loss gets a low score (safe to
prune); one whose removal spikes the loss gets a high score (important,
keep it). This is the most expensive of the four scoring methods but also
the most direct — no approximation, no gradient assumptions, just "does
removing this hurt?"

```python
def compute_kanko_masks(ko_scores, prune_ratio):
    for name, scores in ko_scores.items():
        n_prune = max(1, int(scores.shape[0] * prune_ratio))
        prune_i = torch.argsort(scores)[:n_prune]     # ascending: lowest KO-score first
        mask = torch.ones(scores.shape[0], dtype=torch.bool)
        mask[prune_i] = False
```
Same bottom-`ratio` selection logic as the other three.

### Verifying the soft mask is really equivalent to permanent removal

```python
def apply_kanko_hard_mask(model, masks):
    """Permanently zero pruned channels' weights (in-place)."""
    for name, mask in masks.items():
        module = getattr(model, name)
        prune_idx = (~mask).nonzero(as_tuple=True)[0]
        for pname, p in module.named_parameters():
            if p.dim() >= 1 and p.shape[0] == mask.shape[0]:
                with torch.no_grad():
                    p.data[prune_idx] = 0.0
```
Two different ways to "remove" a channel: the **soft mask**
(`apply_kanko_masks`, a hook that multiplies the output by 0/1 at runtime —
weights untouched) versus this **hard mask** (weights permanently set to
zero, same tensor shape, no hook needed). They should be mathematically
equivalent, and this function exists specifically to prove that — run both,
compare all five metrics, confirm they match to several decimal places. They
did, once the `shuffle=True` test-loader issue was fixed (§5). Neither one
actually shrinks the tensor — that's still only WSAP.

---

## 5. Bugs found and fixed

### 5.1 WSAP — dead-channel collapse from an unbalanced loss
Fine-tuning with a **SAM-only** loss (no MSE term) caused 38 of 128 channels
in a FastKAN layer to collapse to constant zero output — the optimizer found
a shortcut that satisfied the SAM objective by killing the channel entirely,
since SAM is undefined/degenerate for an all-zero vector in a way that
doesn't penalize collapse the way MSE would. **Fix:** switch to
`loss_fn(sr, hr, alpha=0.7)` — the combined SAM+MSE loss shown in §0 — for
every fine-tuning step in every notebook.

### 5.2 Taylor — gradient-collection bugs
- `optimizer.step()` was being called inside the score-collection loop,
  updating the weights while `collect()` was still trying to measure their
  importance — scores were partly computed on a moving target.
- Forward/backward hooks were not removed between pruning ratios in an
  earlier version, so a previous ratio's hooks stayed attached and polluted
  the next ratio's gradient collection. `_remove_hooks()` is now called at
  the end of `collect()` and again in `remove_masks()`.

### 5.3 KAN-KO — masking and evaluation checks
Hard-mask vs. soft-mask comparison (§4) initially disagreed on SAM_M2 (the
batch-mean SAM) by ~0.007, while RMSE, SAM_M1, PSNR, and SSIM all matched
exactly. Root cause: `test_loader` had `shuffle=True`, so the size of the
last (partial) batch changed between the two runs, and SAM_M2 averages
per-batch means rather than per-sample means — a different last-batch size
shifts that average slightly. **Fix:** `shuffle=False` on `test_loader`.
After the fix, all five metrics matched to 6 decimal places.

### 5.4 ShapKAN — missing checkpoint reload
Every other notebook reloads the checkpoint immediately before scoring:
```python
model.load_state_dict(torch.load(CHECKPOINT))   # Taylor: inside the ratio loop
model.load_state_dict(torch.load(CKPT))          # KAN-KO: inside the ratio loop
```
The ShapKAN sweep cell had no equivalent line — it evaluated and scored
whatever `model` object already existed in that Colab session's memory.

**How this was caught:** a diagnostic snapshot compared the in-memory
model's weights against the checkpoint's weights directly:
```python
before = model.fast_kan1.base_conv[0].weight.detach().clone()
baseline_current = evaluate_model_full(model, test_loader, device)   # no reload

model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
after = model.fast_kan1.base_conv[0].weight.detach()
print(torch.allclose(before, after))   # -> False
baseline_reloaded = evaluate_model_full(model, test_loader, device)
```
`torch.allclose` returned `False` with a max absolute weight difference of
~0.13 — the in-memory model genuinely was not the checkpoint. The effect on
the actual numbers:

| | SAM | RMSE |
|---|---|---|
| in-memory model (bug) | 0.1360 – 0.1604 (varied by run) | 0.2781 – 0.2852 |
| freshly reloaded checkpoint | **0.2845** | **0.1391** |

0.2845 / 0.1391 matches Taylor's and KAN-KO's baseline exactly.

**Fix**, placed at the top of the sweep cell, before both the baseline eval
and the Shapley value estimation (the SV estimation was also running on the
wrong model and had to be rerun):
```python
CHECKPOINT = '/content/drive/MyDrive/GKANSU/model_checkpoint.pth'
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.to(device)
model.eval()
```

---

## 6. Architectural constraint — why only WSAP produces a smaller model

`fast_kan1` and `fast_kan2` are both wrapped in a residual connection
(`x = ReLU(fast_kan(x) + x)`), which forces the block's output to have the
same channel count as its input. Slicing channels out of one of these layers
in isolation breaks that addition immediately — shapes stop matching.
**WSAP is the only notebook that builds a bridging mechanism** (the three
1×1 `res_proj`/`restore_proj` convolutions in §2) to work around this, which
is why it's the only one that changes the model's actual parameter count.
Taylor, ShapKAN, and KAN-KO all use the soft-mask approach instead
(multiply the output by 0/1 at runtime) — this correctly tests whether each
method's channel ranking is meaningful, but does not shrink the model.

---

## 7. Results summary

| Method | What changed after fine-tuning | Structural compression? |
|---|---|---|
| Taylor | Near-baseline SAM/RMSE through 40–50%, sharp degradation past 70% | No — sparsity only |
| WSAP | 32.7% parameter reduction, SAM +0.0923° | **Yes** |
| ShapKAN | SAM flat/improved at nearly every ratio; RMSE consistently +0.07–0.09 worse at every ratio (see §3 on the SAM-only value function) | No — sparsity only |
| KAN-KO | Small negative ΔSAM at 30–40%, RMSE degradation modest (0.028–0.033) at those ratios | No — sparsity only |

Full per-ratio tables, formulas, and toy worked examples for all four
scoring methods are in the accompanying paper.
