# Crown Fire Model

This module evaluates whether a surface fire transitions into a crown fire and, if so, computes the resulting crown fire rate of spread and fireline intensity. It implements the Van Wagner (1977) initiation criteria and Rothermel (1991) crown fire spread equations. Crown fire behavior is automatically evaluated by the simulation engine for cells with canopy data — users do not call these functions directly.

::: embrs.models.crown_model
