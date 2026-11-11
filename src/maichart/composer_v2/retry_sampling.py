"""Condition complete-event sampling on known failed completions.

Only exact rejected leaves lose mass. Their ancestors retain the probability
of every untested continuation, including other configurations at the same
time. Log-domain subtree sums avoid 1-p cancellation at low temperatures.
"""
import math
import torch


class RetrySpaceExhausted(ValueError):pass


class _Node:
    __slots__=('children','logits','choices','cpu_logits','blocked','has_blocked','survival','temperature')
    def __init__(self):
        self.children={};self.logits=None;self.choices=();self.cpu_logits=None
        self.blocked=False;self.has_blocked=False;self.survival=None;self.temperature=None
    def __deepcopy__(self,memo):
        from copy import deepcopy
        result=type(self)();memo[id(self)]=result
        result.children=deepcopy(self.children,memo)
        # Model outputs are immutable here. Fork rejection bookkeeping without
        # cloning a GPU tensor for every visited token of every trial branch.
        for name in ('logits','choices','cpu_logits','blocked','has_blocked','survival','temperature'):
            setattr(result,name,getattr(self,name))
        return result


class RejectedEventTree:
    def __init__(self):
        self.root=_Node();self.rejections=0
    def _path(self,prefix,create=False):
        path=[self.root];node=self.root
        for token in prefix:
            if token not in node.children:
                if not create:return None
                node.children[token]=_Node()
            node=node.children[token];path.append(node)
        return path
    def cached(self,prefix):
        path=self._path(prefix)
        return None if path is None else path[-1].logits
    def remember(self,prefix,choices,logits,*,temperature=None):
        path=self._path(prefix,True);node=path[-1]
        choices=tuple(sorted(set(choices)))
        if node.logits is None:node.logits=logits.detach()
        if node.choices!=choices or node.temperature!=temperature:
            node.choices=choices;node.temperature=temperature
            for parent in path:parent.survival=None
        return node.logits
    def reject(self,completion):
        node=self.root;path=[node]
        for token in completion:
            # Fixed-count alternative proposals can omit a terminal model
            # query. They enter the physical memo but are conditioned out only
            # once the free-event sampler has evaluated their complete path.
            if node.logits is None or token not in node.choices:return False
            node=node.children.setdefault(token,_Node());path.append(node)
        if node.blocked:return False
        node.blocked=True;self.rejections+=1
        for parent in path:parent.has_blocked=True;parent.survival=None
        missing=[n for n in path[:-1] if n.cpu_logits is None]
        if missing:
            # One transfer per failed proposal, none on normal valid draws.
            rows=torch.stack([n.logits for n in missing]).float().cpu().double()
            for i,parent in enumerate(missing):parent.cpu_logits=rows[i]
        return True
    def _survival(self,node,temperature):
        if not node.has_blocked:return 0.
        stack=[(node,False)]
        while stack:
            current,expanded=stack.pop()
            if current.survival is not None:continue
            if current.blocked:current.survival=-math.inf;continue
            needed=[child for child in current.children.values()
                    if child.has_blocked and child.survival is None]
            if needed and not expanded:
                stack.append((current,True));stack.extend((child,False) for child in needed)
                continue
            adjustments=[]
            for token in current.choices:
                child=current.children.get(token)
                adjustments.append(child.survival if child is not None and child.has_blocked else 0.)
            local_temperature=temperature if current.temperature is None else current.temperature
            if local_temperature==0:
                # A greedy draw can still take the best surviving choice.
                current.survival=0. if any(math.isfinite(value) for value in adjustments) else -math.inf
                continue
            values=current.cpu_logits[list(current.choices)]/local_temperature
            remaining=values+torch.tensor(adjustments,dtype=torch.float64)
            current.survival=float(torch.logsumexp(remaining,0)-torch.logsumexp(values,0))
        return node.survival
    def condition(self,prefix,choices,temperature):
        """Return logits already scaled by temperature and surviving choices."""
        node=self._path(prefix)[-1]
        if not node.has_blocked:return node.logits,tuple(choices),False
        adjustments=[];allowed=[]
        for token in sorted(set(choices)):
            child=node.children.get(token)
            survival=self._survival(child,temperature) if child is not None and child.has_blocked else 0.
            if math.isfinite(survival):allowed.append(token);adjustments.append(survival)
        if not allowed:raise RetrySpaceExhausted('No untried complete event remains in this retry subtree')
        if temperature==0:return node.logits,tuple(allowed),False
        # Add in FP64 before casting: a near-certain failed leaf can require
        # cancellation of very large log odds without losing the alternatives.
        base=node.cpu_logits if node.cpu_logits is not None else node.logits.detach().float().cpu().double()
        values=base[allowed]/temperature+torch.tensor(adjustments,dtype=torch.float64)
        values-=values.max()
        adjusted=torch.zeros_like(node.logits,dtype=torch.float32)
        adjusted[torch.tensor(allowed,device=adjusted.device)]=values.to(adjusted.device,adjusted.dtype)
        return adjusted,tuple(allowed),True
