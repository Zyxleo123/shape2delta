"""shape2delta: does a drug target's conformational ensemble carry transferable
signal for transcriptional drug-response prediction?

Feature tiers are strictly nested: T0 (identity) ⊂ T1 (static structure) ⊂ T2
(ensemble). The headline quantity is always the T2−T1 gap, never absolute score.
"""
__version__ = "0.0.1"
