"""Paired diagnostics on held-out, positively supervised geometry decisions."""
import torch


class SpatialValidation:
    def __init__(self, model, batches):
        self.history = hasattr(model, 'spatial_history')
        # Deterministic positions in the existing validation order.
        self.probes = {round(i * (batches - 1) / 2) for i in range(3)} if self.history and batches else set()
        self.buttons = model.rim_button_ids.detach().cpu()
        self.kinds = torch.tensor([model.codec.ids[k] for k in ('N:tap', 'N:slide', 'N:hold') if k in model.codec.ids])
        self.count = self.changed = self.gained = self.lost = 0
        self.probability_shift = 0.
        self.example = None
        self.geometry_sums = None

    def queries(self, batch, index):
        if index not in self.probes:
            return None
        eligible = (batch['token_weights'] > 0) & batch['token_mask'] & (batch['sample_weight'][:, None] > 0)
        mask = eligible & torch.isin(batch['input_ids'], self.kinds) & torch.isin(batch['labels'], self.buttons)
        return mask.nonzero(as_tuple=True) if mask.any() else None

    def compare(self, model, batch, cache, queries):
        rows = tuple(index.to(batch['input_ids'].device) for index in queries)
        options = dict(incremental=False, button_queries=rows)
        off_probs = model.decode(batch, cache, use_spatial_history=False, **options).float().softmax(-1)
        on_probs = model.decode(batch, cache, **options).float().softmax(-1)
        off, on = off_probs.argmax(-1), on_probs.argmax(-1)
        off, on = model.rim_button_ids[off], model.rim_button_ids[on]
        truth = batch['labels'][rows]
        changed = on != off
        differences = torch.stack((changed.sum(), ((on == truth) & changed).sum(),
                                   ((off == truth) & changed).sum())).tolist()
        self.count += len(truth)
        self.changed += differences[0]
        self.gained += differences[1]
        self.lost += differences[2]
        self.probability_shift += float((off_probs - on_probs).abs().sum()) / 2
        if differences[0] and self.example is None:
            i = changed.nonzero()[0, 0]
            keys = torch.stack((off[i], on[i], truth[i])).tolist()
            self.example = dict(zip(('off', 'on', 'authored'), (model.codec.tokens[k][2:] for k in keys)))

    def geometry(self, batch):
        weights = batch['slide_geometry_weights'].float()
        targets = batch['slide_geometry_targets'].float()
        totals = torch.cat((weights.sum()[None], (targets * weights[:, None]).sum(0)))
        self.geometry_sums = totals if self.geometry_sums is None else self.geometry_sums + totals

    def report(self, numerator, rows):
        report = {}
        if self.history:
            report['history'] = dict(count=self.count, changed=self.changed, gained=self.gained,
                                     lost=self.lost, example=self.example,
                                     mean_tv=self.probability_shift/self.count if self.count else None)
        if self.geometry_sums is not None:
            weight, *sums = self.geometry_sums.tolist()
            if weight > 0:
                means = torch.tensor(sums, dtype=torch.float64).div(weight).clamp(0, 1)
                # Entropy gives the lowest BCE any fixed per-field prediction
                # could achieve on these same weighted validation targets.
                entropy = -(torch.special.xlogy(means, means) + torch.special.xlogy(1 - means, 1 - means))
                report['slide'] = dict(rows=rows, bce=float(numerator)/weight, constant_bce=entropy.mean().item())
        return report
