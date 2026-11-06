# drift

run the same neural network on the same input twice and you can get different numbers. run it on a cpu instead of a gpu, or with 32 images in a batch instead of 1, and you reliably do. this repo measures when two runs of the same model stop agreeing, how far apart they get layer by layer, and whether it ever changes the prediction. the model is resnet18 on the cifar-10 test set (all 10,000 images), with a small transformer on ag news as a second opinion. every number below comes from a run whose json sits beside it in `results/`; nothing is typed in by hand.

![divergence by layer](results/LABEL/depth.png)

<!-- HEADLINE -->

## what is measured

one trained model, one fixed reference run, and one knob changed at a time:

- batch size 1, 8, 32, 128: each image's logits computed alone against the same image inside a batch
- cpu against gpu
- tf32 on against off (`torch.backends.cuda.matmul.allow_tf32` and `torch.backends.cudnn.allow_tf32`)
- `torch.backends.cudnn.benchmark` and `torch.backends.cudnn.deterministic`
- `torch.use_deterministic_algorithms(True)` with `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- cpu thread count 1, 4 and the default
- the identical configuration run again, in the same process and in a new process

for every pair: max and median absolute difference in the logits, relative difference (per image, `||a - b|| / ||a||`), how many images came out bitwise identical, how many changed their top-1 class, and where the changed ones sit in the reference run's margin distribution (the margin is the gap between the top two logits). forward hooks on every leaf module record the per-layer absolute and relative difference for a subset of inputs. wall-clock per configuration is recorded so the cost of each knob is visible.

## results

<!-- RESULTS -->

## what the numbers say

<!-- SAY -->

## what to actually do about it

- do not chase bitwise reproducibility across batch sizes, devices or cudnn's autotuner. the forward pass of a trained network is not the same computation at batch 1 and batch 32, and no flag makes it so. compare models and predictions with a tolerance, and pick the tolerance from a measurement like the tables above rather than from a feeling.
- if you need the same bits from the same configuration twice, you already have it: the default kernels at a fixed shape were bitwise repeatable here, in the same process and across processes. `torch.use_deterministic_algorithms(True)` buys you nothing in that setting and costs a little; it earns its keep for ops with atomics (scatter, index_add, some backward kernels), which an inference-only resnet does not touch.
- if you do turn determinism on, measure what it costs on your shapes. here it was a few percent of wall-clock, not the 2x people warn about, because resnet18's convolutions already run deterministic cudnn kernels.
- set `torch.backends.cudnn.benchmark` to the same value in every run you intend to compare, and keep the comparison on one device. those two knobs moved the logits more than anything else.
- tf32 is a non-event on turing (t4) because the hardware has no tf32 units. on ampere and later it is on by default for cudnn convolutions, so a run on a t4 and a run on an a100 differ for a reason that has nothing to do with the code. turn it off explicitly (`torch.backends.cudnn.allow_tf32 = False`) if cross-machine agreement matters.
- the quantity to watch is the margin between the top two logits, not the logit error. a flip needs the disagreement to cross the margin. report the smallest margin in your evaluation set next to the largest disagreement your setup produces; if there is an order of magnitude between them, flips are not your problem.
- what would make flips appear is a low-margin regime: a weaker model, a harder dataset, lower precision (fp16, bf16, int8) or a decision threshold placed near the bulk of the score distribution. that is where the measurement should be repeated before trusting a deployment.

## reproducing it

```
pip install -r requirements.txt
bash run.sh <label>          # results/<label>/: json, charts, summary.md
```

`drift_gpu.ipynb` does the same on a colab gpu. without a gpu the sweep runs the cpu knobs only. the weights are committed in `weights/` with the training log beside them (`python -m drift.train resnet` and `python -m drift.train text` retrain them with the same seeds).

## where it falls short

- one model family for the headline (resnet18), one dataset (cifar-10), one gpu (a colab t4, turing). the transformer is small and trained for three epochs on a laptop; it is a second opinion, not a second study.
- the t4 cannot test tf32 at all. the row is there so the code path is exercised; the number says nothing about ampere.
- no training, no distributed, no mixed precision beyond the tf32 flag, no quantisation. the knobs that move logits by 1e-2 rather than 1e-5 are all on that list.
- margins here are wide because the model is good. a 70% model on a harder set would put more images near zero margin, and the flip rate would stop being zero. the method applies; the headline does not transfer.
- the per-layer curves use 64 inputs, chosen with a fixed seed, not the whole test set.
- the cpu runs use whatever cores the colab vm has (two), so the thread-count knob is 1 against 2, and the laptop this was built on was too throttled to add a 4-thread row in time.
- the resnet weights were fetched, not trained here (`weights/resnet18_cifar10.json` has the source and checksum). the normalisation constants in `drift/data.py` are the ones that checkpoint expects, which are not the cifar-10 values most tutorials use.
