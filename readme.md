# From Pixels to Prepositions

This code is part of the research done at IIT Madras. 

## Abstract
Human language is influenced by sensory-motor experiences. Sensory experiences gathered in a spatiotemporal world are used as raw material to create more abstract concepts. In language, one way to encode spatial relationships is through spatial prepositions. Spatial prepositions that specify the proximity of objects in space, like far and near or their variants, are found in most languages. The mechanism for determining the proximity of another entity to itself is a useful evolutionary trait. From the taxic behavior in unicellular organisms like bacteria to the tropism in the plant kingdom, this behavior can be found in almost all organisms. In humans, vision plays a critical role in spatial localization and navigation. This computational study analyzes the relationship between vision and spatial prepositions using an artificial neural network. For this study, a synthetic image dataset was created, with each image featuring a 2D projection of an object placed in 3D space. The objects can be of various shapes, sizes, and colors. A convolutional neural network is trained to classify the object in the images as far or near based on a set threshold. The study mainly explores two visual scenarios: objects confined to a plane (grounded) and objects not confined to a plane (ungrounded), while also analyzing the influence of camera placement. The classification performance is high for the grounded case, demonstrating that the problem of far/near classification is well-defined for grounded objects, given that the camera is at a sufficient height. The network performance showed that depth can be determined in grounded cases only from monocular cues with high accuracy, given the camera is at an adequate height. The difference in the network’s performance between grounded and ungrounded cases can be explained using the physical properties of the retinal imaging system. The task of determining the distance of an object from individual images in the dataset is challenging as they lack any background cues. Still, the network performance shows the influence of spatial constraints placed on the image generation process in determining depth. The results show that monocular cues significantly contribute to depth perception when all the objects are confined to a single plane. A set of sensory inputs (images) and a specific task (far/near classification) allowed us to obtain the aforementioned results. The visual task, along with reaching and motion, may enable humans to carve the space into various spatial prepositional categories like far and near. The network’s performance and how it learns to classify between far and near provided insights into certain visual illusions that involve size constancy.

## Research Paper

**Authors:** Raj S R, K., Chakravarthy V, S. & Sahoo
**Title:** From Pixels to Prepositions: Linking Visual Perception with Spatial Prepositions Far and Near
**Journal:** Cognitive Computation
**Year:** 2024
**Link:** [[Link](https://doi.org/10.1007/s12559-024-10329-6)]

## BibTeX Citation

If you want to cite this project in your research paper, you can use the following BibTeX entry:

```bibtex

@article{Raj_S_R2024,
  title={From Pixels to Prepositions: Linking Visual Perception with Spatial Prepositions Far and Near},
  author={Raj, S R, Krishna and Chakravarthy, V, Srinivasa and Sahoo, Anindita},
  journal={Cognitive Computation},
  year={2024},
  doi={10.1007/s12559-024-10329-6},
  url={https://doi.org/10.1007/s12559-024-10329-6},
  issn={1866-9964},  
}
```
## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

The code base includes the model and the training code.
The analysis,logging and visualization code is not currently included. 


### Requirements

- Python 3.x
- Keras
- Pandas
- TensorFlow
- NumPy

