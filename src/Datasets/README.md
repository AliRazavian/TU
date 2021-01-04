Datasets and Experiment_set
=========

**Dataset** is an object that is defined to store a full set of data samples. These samples are organized into different __Experiment_set__. __Experiment_set__ could be ``train``, ``val``, ``test`` or any other name. A __Task__ defines what kind of an experiment should be defined for an __Experiment_set__.

An __Experiment_Set__ is a matrix of *data samples*. A data sample is a single set of information with multiple modalities. Although it is NOT defined as a matrix, rather as a list of modalities, and each modality is an independent column.

Any __Experminet_Set__ should have the following behavior:
1. __\_\_init\_\_(global\_cfgs,dataset\_name)__ should parse the JSON file and create appropriate the modalities
2. batch = __get\_next_batch__() should return a batch of data where the dependencies between modalities are met. Also, the dataset requirements are satisfied
3. len = __len()__ returns the number of samples


**Modalities** depend on one another. The process of learning is to establish causality or correlation between different modalities.

Batch behaviour
---------------

The `batch` is a dictionary that gets updated by the different modalities. Each modality has `get_batch_name()` that defines its entry
point to the `batch` dictionary, e.g. a label modality with the name `fracture` would add to the `batch`:

```python
batch = {
  "Image": tensor,
}

myFracture.get_batch(batch)

batch == {
  "Image": tensor,
  ['target_%s' % (myFracture.get_name())]: data,
}
```

In the above the `get_batch_name` is responsible for the `'target_%s' % (myFracture.get_name())` and this
is defined in `base_modality` but overridden for certain subclasses, e.g. `base_output`. The default is that
the name is `encoder_` for the `encoder` detects it.
