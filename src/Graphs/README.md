# Graph

**Graph** is an abstract object that maps the relationship between the input and output modalities. the most simple form of graph could be just a deep convolution network mapping the input to the output

```mermaid
graph LR
	image((image)) --> |4x1x1x256x256|image_path[image<br/>path]
	subgraph image_reconstruction
		image
	end
	implicit_image((implicit_image)) --> |4x1x256x16x16|visual_path[visual<br/>path]
	subgraph Exam_View_classification
		Exam_View
	end
	subgraph Exam_Type_classification
		Exam_Type
	end
	subgraph Body_Part_classification
		Body_Part
	end
	subgraph Fracture_classification
		Fracture
	end
	subgraph Fracture_displaced_classification
		Fracture_displaced
	end
	subgraph Implant_hemi_knee_classification
		Implant_hemi_knee
	end
	implicit_Exam_View((implicit_Exam_View)) --> |4x1x16x2x2|Exam_View_path[Exam<br/>View<br/>path]
	implicit_Exam_Type((implicit_Exam_Type)) --> |4x1x32x2x2|Exam_Type_path[Exam<br/>Type<br/>path]
	implicit_Body_Part((implicit_Body_Part)) --> |4x1x32x2x2|Body_Part_path[Body<br/>Part<br/>path]
	implicit_Fracture((implicit_Fracture)) --> |4x1x8x2x2|Fracture_path[Fracture<br/>path]
	implicit_Fracture_displaced((implicit_Fracture_displaced)) --> |4x1x8x2x2|Fracture_displaced_path[Fracture<br/>displaced<br/>path]
	implicit_Implant_hemi_knee((implicit_Implant_hemi_knee)) --> |4x1x8x2x2|Implant_hemi_knee_path[Implant<br/>hemi<br/>knee<br/>path]
	implicit_fork((implicit_fork)) --> |4x1x1024x2x2|fork[fork]
	Body_Part_path --> |4x1x23|Body_Part((Body<br/>Part))
	Exam_Type_path --> |4x1x31|Exam_Type((Exam<br/>Type))
	Exam_View_path --> |4x1x15|Exam_View((Exam<br/>View))
	Fracture_displaced_path --> |4x1x1|Fracture_displaced((Fracture<br/>displaced))
	Fracture_path --> |4x1x1|Fracture((Fracture))
	Implant_hemi_knee_path --> |4x1x1|Implant_hemi_knee((Implant<br/>hemi<br/>knee))
	fork --> |4x1x16x2x2|implicit_Exam_View((implicit<br/>Exam<br/>View))
	fork --> |4x1x32x2x2|implicit_Exam_Type((implicit<br/>Exam<br/>Type))
	fork --> |4x1x32x2x2|implicit_Body_Part((implicit<br/>Body<br/>Part))
	fork --> |4x1x8x2x2|implicit_Fracture((implicit<br/>Fracture))
	fork --> |4x1x8x2x2|implicit_Fracture_displaced((implicit<br/>Fracture<br/>displaced))
	fork --> |4x1x8x2x2|implicit_Implant_hemi_knee((implicit<br/>Implant<br/>hemi<br/>knee))
	image_path --> |4x1x256x16x16|implicit_image((implicit<br/>image))
	visual_path --> |4x1x1024x2x2|implicit_fork((implicit<br/>fork))

```

## Model

**Model** is an object that maps two **modalities** together.
A **Model** must have the following objects:

1. `encoder` and `decoder` are the `neural_nets` that does the mapping
2. `heads` and `tails` are the head and tail **modalities**
3. `factory` is a factory that create singleton `encoder` and `decoder`. The reason that these neural netowrks are signleton is that we want to make sure that different graphs and tasks share the same mappings between modalities. (for example, during the train and test, we should have the same neural networks do the mappings)

**Model** also has the following functions:

1. `__init__` initializes the model based on `model_cfgs`, `graph_cfgs`, `scene_cfgs` and `scenario_cfgs`.
2. `encode(batch)` and `decode(batch)` forwards the `batch` into `encoder` and `decoder` neural networks, which in turn, calculate the results and put it back in the `batch`. `batch` is a dictionary that stores all the modality `tensors`, `loss` values, `results` and `times` in it.
3. `update_modality_dims()` is an important function that automatically calculate the dimensions of the modalities if they are not specified in other config files. The reason that we calculate the modality dimension on the fly is that this way, we can dynamically make the graph and don't be bothered with the input,output sizes of tensors.
4. `step()`, `zero_grad()`, `update_learning_rate(learning_rate)`, `train()` and `eval()` are calling the same functionalities in the `encoder` and `decoder` neural networks, if they exists.

## Factories

**Factory** is a _singleton_ object that generates neural networks.
There are multiple categories of factories:

1. **Cascade_factory** is responsible for generating cascades of neural networks.
