This folder contains all the definitions used. Everything is in yaml format although jsons are allowed as well.

# Datasets

This folder contains the datasets available. Each dataset defines what outcomes and what columns should be used from a corresponding CSV-file.

# Graphs

Each task has a graph that is set of neural networks that combine the full network. This is the core of the network but the dataset usually also contain information about the bottom/top layers.

# Scenarios

A scenario is a grouping of different tasks. You can start with a completely unregularized network and then as we progress we can add/remove regularizers such as auto-encoders, pseudo labels etc.

# Tasks

For each scene in a scenario you can have one or more tasks. These contain inforation about the regularizers, batch size multipliers and more. These can be over-ridden by adding a `task_defaults` in the scenario's `scene` config. Tasks may also share configurations by having a `template` key which refers to a template in the `Tasks/Template` folder.
