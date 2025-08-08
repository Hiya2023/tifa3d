# TIFA3D: Faithfulness Evaluation for Text-to-3D Generation

## Description

**TIFA3D** is an extension of [TIFA](https://github.com/Yushi-Hu/tifa), a SOTA metric for evaluating text-to-image generation. TIFA3D adapts this framework for **3D object evaluation**.

It evaluates how well a generated 3D object matches a given text prompt using a vision-language QA approach. The process includes:

- Rendering 12 azimuthal views uniformly distributed over a 360° rotation around the object.
- Using **Language model** to generate candidate question–answer pairs for each text prompt.
- Filtering those questions using UnifiedQA model.
- Passing each view and questions through a **VQA model** (e.g., mPLUG).
- Aggregating the scores to compute the final TIFA3D score for a given text and 3D object pair.

## Running the Project

### Step 1: Compute per-prompt scores

Run `tifa_composite.py` to compute faithfulness scores for all prompt-model-aspect combinations

### Step 2: Post-process final scores

- After computing tifa scores for each prompt, aggregate them with `post_processing_score.py`
- Getting a tifa aggregated score for each t23d model- aspect pair.

## Citation
```text
@inproceedings{hu2023tifa,
  title={Tifa: Accurate and interpretable text-to-image faithfulness evaluation with question answering},
  author={Hu, Yushi and Liu, Benlin and Kasai, Jungo and Wang, Yizhong and Ostendorf, Mari and Krishna, Ranjay and Smith, Noah A},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={20406--20417},
  year={2023}
}
```


