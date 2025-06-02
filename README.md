# UniVLN: Universal Vision-Language Navigation

## 📚 Data

The necessary data can be downloaded from [HuggingFace](https://huggingface.co/datasets/JunweiZheng/UniVLN)

## 📦 Training

The baselines we used are zero-shot MLLM so it's not necessary to train the model.

## 📦 Evaluation

We benchmark zero-shot OpenFMNav, SG-Nav and UniGoal. The code repositories of the baselines can be found in this UniVLN repository. Since we use Habitat-Sim and Habitat-Lab for simulation, which are the same as the three baseline model code repositories, please refer to those repositories for more details.

As for the modalities integrated in Habitat, we strongly recommend checking this [Documentation](https://aihabitat.org/docs/habitat-sim/). For the modalities that are not integrated in Habitat, we provide a code snippet about how to get the inputs.
