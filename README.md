# DeepMed

This software is designed to learn the causality/correlation between different modalities of medical data.
These modalities include Numbers, signals with 1D consistency like language or ECG, signals with 2D consistency like images or activation maps and signals with 3D consistency like videos or MRI.

## Installation

This software is written in python language version 3.6 & 3.7
See the `requirements.txt` for a package dump. To insall you can use `conda install --file requirements.txt`
or install manually the following packages:

[pandas:](https://pandas.pydata.org/) for handling dataframes and CSVs.

`pip install pandas --user`

[scikit-image:](http://scikit-image.org/) for image operations

`pip install scikit-image --user`

[opencv:](https://opencv.org/) for image operations

`pip install opencv-python --user`

At the moment, we use [_pytorch_](https://pytorch.org/) back-end for our neural network operations

## Important structures

The main objective of this software is automatically to create a _Graph_ between different modalities defined in _Datasets_, update the parameters of the _Graph_ based on the samples in the _Dataset_ and evaluate the results (training).

## Sequence for calling train

![README](https://www.plantuml.com/plantuml/png/VPD1Rzim38Nl_XM4Usa33BaRoD1X0uPTszu5LkPijRPaIFIszjUFfD8qJeVb45ZIu_UeP_gsC6PUvwdvuZbbc374Pf6juCE2aU7UKyGSqeTDskmyO7dUR7pFk8nbi_KD6GjnqnmEy2C63ZY5L-IG8XmEZJh3fyydmmvyYeNpgdrTtpHJIWiG-X6AkSD9UL6L2Pcgg2ZZPnXAGx4ttuSiDY4EwcX47dDQX_4T5fWoYkM5WRRNGTiwzoPmhcr559EuvJ2CB0iGxcvNyu7q-4IUORSJBpswWg-rj_DZPMukQb0LQqH5-YD0_-Ad5m-3Pge5SWLsjZhJ6zpqS-ffAYTXNJgClSF-QItiIdYbSYoLPEmEydZ-2gY3l4PdCKhw09RMtHfhiWhFjAZZ3cX9VYnj2nncAjPElQAmT30wM9TUBpuHPajV0YJJ6VKHMLvhpLMdeMsDQ6rkg1-uXMcnxlHtdA_tqqX_-Fe61f8SpUCer5DLrHdjW7KeH2ZMzkY4uYBEjKKTdNzOt-ClWj-1H_1hpYJpRS70Vg-L-ttpLdbH9wa2Ks7eZv_3VyYLg3p1JbXOi0gzbJ_vPly2 'README')

```plantuml
@startuml
Actor main
participant Scenario
participant Scene
participant Task
participant Graphs
participant Dataset

== Initialization ==
main -> Scenario: constructor()
loop each scene
  Scenario -> Scenario: create scenes configs
end

== Training ==
loop through Scenario iterator
  Scenario -> main: ""__iterator__""
  note right
    ==** Scene object **==
    # Create Scene object: ""constructor()""
    # Inits the acyclyc graph: ""init_graph()""
    # Creat the models: ""init_models_and_adjust_sizes()""
  end note
  main -> Scene: run_scene()
  loop Run scene: ""repeat * epochs"" times
    Scene -> Task: update learning rate
    loop batch iteration ""epoch_size"" times
      Scene -> Task: step()
      Dataset -> Task: next() gets batch
      Task -> Graphs: train() on train_set_name
    end
    Scene -> Task: Save with current scene //name//
  end
  Scene -> Task: run test dataset
  Scene -> Task: Save with scene name //last//
end
@enduml
```

## Tests

In the `tests` folder you will find component tests. You can run all of them using `nose2` (install `conda install -y nose2`)
