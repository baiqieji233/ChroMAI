# Training Epoch Semantics

The WebUI `epochs` field has two meanings depending on the training mode:

- `fresh`: total epochs in the new training stage, starting at epoch 0.
- `finetune`: total epochs in the new fine-tuning stage, starting at epoch 0.
- `resume`: additional epochs after the checkpoint's saved epoch.

For example, resuming checkpoint epoch 4 with `epochs=3` runs epochs 5, 6, and
7. The task status reports `epoch_start=4`, `epochs_requested=3`, and
`epoch_target=7`; progress uses the absolute target so it does not reach 100%
after the first resumed epoch.

The WebUI Python server reads `server.py` when it starts, so backend epoch
metadata changes require a server restart. Static files are read for each HTTP
request with `Cache-Control: no-store`; refreshing an existing page loads the
updated epoch display JavaScript without restarting the server. An older running
task has no `epoch_target`, so the refreshed page hides its percentage bar and
shows the last logged epoch with an unknown target.
