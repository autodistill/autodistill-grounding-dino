<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/autodistill/autodistill-banner.png?3"
      >
    </a>
  </p>
</div>

# Autodistill Grounding DINO Module

This repository contains the code supporting the Grounding DINO base model for use with [Autodistill](https://github.com/autodistill/autodistill).

[Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) is a zero-shot object detection model developed by IDEA Research. You can distill knowledge from Grounding DINO into a smaller model using Autodistill.

Read the [Grounding DINO Autodistill documentation](https://autodistill.github.io/autodistill/base_models/grounding-dino/).

> [!TIP]
> You can use Autodistill Grounding DINO on your own hardware, or use the [Roboflow hosted version of Autodistill](https://blog.roboflow.com/launch-auto-label/) to label images in the cloud.

## Installation

To use the Grounding DINO base model, you will need to install the following dependency:

```bash
pip3 install autodistill-grounding-dino
```

## Quickstart

```python
from autodistill_grounding_dino import GroundingDINO
from autodistill_yolov8 import YOLOv8


# define an ontology to map class names to our GroundingDINO prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = GroundingDINO(ontology=CaptionOntology({"shipping container": "container"}))

# label all images in a folder called `context_images`
base_model.label("./context_images", extension=".jpeg")
```

## License

The code in this repository is licensed under an [Apache 2.0 license](LICENSE).

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!
