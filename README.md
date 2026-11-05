# drift

run the same neural network on the same input twice and you can get different numbers. run it on a cpu instead of a gpu, or with 32 images in a batch instead of 1, and you reliably do. this repo measures when two runs of the same model stop agreeing, how far apart they get layer by layer, and whether it ever changes the prediction. the model is resnet18 on the cifar-10 test set (all 10,000 images), with a small transformer on ag news as a second opinion. every number below comes from a run whose json sits beside it in `results/`; nothing is typed in by hand.

![relative divergence by layer](results/LABEL/depth.png)

<!-- INTRO -->

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

<!-- TABLES -->

## reproducing it

```
pip install -r requirements.txt
bash run.sh <label>          # results/<label>/: json, charts, summary.md
```

`drift_gpu.ipynb` does the same on a colab gpu. without a gpu the sweep runs the cpu knobs only. the weights are committed in `weights/` with the training log beside them (`python -m drift.train resnet` and `python -m drift.train text` retrain them with the same seeds).

## where it falls short

<!-- SHORT -->
