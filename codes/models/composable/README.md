This directory contains the code for my vision for how building machine learning models should actually work: they should be 
composable models from primitives that are defined by your inputs and outputs, and agnostic to anything else.

Building a composable model requires you to make a few decisions:

1. What inputs can you provide? More is always better.
2. What type of outputs do you expect?
3. **Very roughly** how much compute do you want to throw at the problem?

## Some basic concepts

### Structure

Before we go much further, I want to define an important notion of "structure" which will be used throughout this document.
"Structure" refers to how wide the aggregate variable dimensions of your input and outputs would be.

For example, a 256x256px input image has a "structural" dimension of 256*256=65,536.

I use three classifications for structure in this document. 

1. **Pointwise** has a structural dimension = O(1).
2. **Low structure** data has a structural dimension < O(1000) *(at least for us mere mortals)*
3. **Highly structured** data has a structural dimension > O(1000) magnitude.

These definitions are bounded by power laws. Dense computation in machine learning is essential for good performance,
but consumes O(N²) compute and memory. It is therefore impossible to perform dense computation on highly structured data,
and we must reduce it first. Composable models takes care of this for you, as long as you make the distinction on what
input types you provide.

### Alignment


## Building a composable model

Composable models are built by simply defining your inputs, your outputs, and the compute you wish to expend. 

Lets come up with a fairly preposterous toy problem to show the power of composable models. Say you have a dataset 
consisting of pictures of animals, a textual description of those animals, a label for each animal and an audio clip
of the sounds that animal was making when the picture was taken. You want to build a model that predicts the sound. Here
is how you would do it: 

```python
image_input = Input(structure=HighStructure, dimensions=2, compute=Medium)
text_input = Input(structure=LowStructure, dimensions=1, discrete=True, compute=High)
class_input = Input(structure=Point, discrete=True, compute=Medium)

sound_output = Output(structure=HighStructure, dimension=1)

model = UniversalModel(unaligned_inputs=[image_input, text_input, class_input], aligned_inputs=[], outputs=[sound_output], compute=Medium)
```


Once you've decided these, you can build a composable model.

There are four "types" of composable models:
1. Fan-in models, which reduce a highly structured input (e.g. images) to a less structured output. Applications are classifiers, coarse object detection and speech recognition.
2. U models, which efficiently process highly structured inputs. Applications are generative models and fine object detection.
3. Straight models, which perform dense computation on less structured data. Think text inputs and outputs. Applications are text generation
4. Fan-out models, which take low-structured inputs and produce highly structured outputs. Applications are generative networks like GANs, though I recommend you actually use diffusion models for this purpose.

A composable model consists of three parts: