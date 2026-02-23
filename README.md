# Computer_Vision_Exam
This work focuses on analyzing the gain in performance of backwards compatible models as shown in the paper [Towards Backward-Compatible Representation Learning](https://arxiv.org/abs/2003.11942). The implementation focuses on confirming the compatibility of models by using equation (9) of the paper: \
$G(\phi_{new}, \phi_{old}; \mathcal{Q}, \mathcal{D})=\frac{M(\phi_{new}, \phi_{old}; \mathcal{Q}, \mathcal{D})-M(\phi_{old}, \phi_{old}; \mathcal{Q}, \mathcal{D})}{M(\phi_{new}^*, \phi_{new}^*; \mathcal{Q}, \mathcal{D})-M(\phi_{old}, \phi_{old}; \mathcal{Q}, \mathcal{D})}$

## Implementation

|  Exercise   | DONE  | WIP | Link |
|-----|---|---|---|
| Main architecture | ✅ | | [Go to section](#main-architecture)|
| Evaluation metrics | ✅ | | [Go to section](#evaluation-metrics)|
| MNIST-1D tests | ✅ | |[Go to section](#mnist-1d)|
| CIFAR-100 tests |  | 📝 |[Go to section](#cifar-100)|
| MEDMNIST tests |   | 📝 |[Go to section](#medmnist)|
| Inverted models |   | 📝 |[Go to section](#inverted-models)|
| Dimension expansion |   | 📝 |[Go to section](#dimension-expansion)|

## File Structure

The root folder for these project contains the following:

- _src_: contains code to run each experiment individually
- _demo.pynb_: is the demo notebook that shows what I achieved during my tests
- _requirements.txt_: a file with all the necessary dependencies to run the code

## Environment

All labs have been developed and tested in this environment and it can be recreate by running the following commands:

```linux

python -m venv CV_CICCHESE
source CV_CICCHESE/bin/activate

pip install -r requirements.txt
```
## MAIN ARCHITECTURE

### Models
For each task two classes of models have been defined: _OldNet_ and _NewNet_. In all the tests _OldNet_ is a less powerful newtwork that represents an old architecture that we wish to improve on, at the same time _NewNet_ is the new improved network that we would like to use as an improvement. _NewNet_ is a more powerful model that, after training, should be abe to be replaced in place of the old one with minimal disruption.

### Training
The training strategy follows what is described in the paper to make the new model compatible with the old one by applying a correction to the loss of the new model that penalizes it if its embedding is not compatible with the old one. All new models have also been trained on more classes that the old models.

## EVALUATION METRICS
The gain equation explained in the paper proposes a framework for evaluating models but leaves flexibility of choice on the actual metric to use in the evaluation of these models. I chose to test the models on three metrics that I consider relevant for the chosen datasets.

### Top1 accuracy

### Mean average precision

### Protorype accuracy

## MNIST-1D
MNIST-1D is a simple dataset that has been used to validate and test the code before applying it to more challenging datasets.

## CIFAR-100

## MEDMNIST

## INVERTED MODELS

## DIMENSION EXPANSION