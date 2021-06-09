# tarsius

This repository contains the data and scripts that's necessary to reproduce the results in paper "Evaluating Saliency Methods for Neural Language Models" published in NAACL 2021.

If you are only curious about the evaluation data we used, check out the `data` repository. The scripts used to prepare these data are in `scripts/data-prep/prepare_{number,conll,winobias}`.

If you need to reproduce our analysis, please follow the instructions below.

## Preparation

Check out this repository and the two submodules (`awd-lstm-lm` and `fairseq`).

`cd` into `data/`, download and unzip the [Wikitext-103 dataset](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip) dataset. Both `awd-lstm-lm` and `fairseq` need that dataset to build dictionaries. This process is automatic for `awd-lstm-lm` but needs to be run separately for `fairseq`, so also provide the [preprocessed wikitext-103 binary data dump](http://cs.jhu.edu/~sding/downloads/naacl2021/data-bin/wikitext-103.tar.gz) for fairseq.

`cd` into `models/` and follow the `README` to download finetuned models with `number` or `gender` prediction head.

## Generate Saliency Output and Evaluate Plausibility

Follow the script in `scripts/plausibility`. Note that you need to run `run_awd.sh` for LSTM and QRNN models and `run_fairseq` for Transformer models.

Your output should be the following series of files:

```
out  out.malum  out.overall  out.passed.idx  out.pos.idx  out.verum
```

+ `out.overall`: shows the overall statistics of plausibility test. Those are the numbers we put in our tables for plausibility results.
+ `out.pos.idx`: a list of example indexes that fell into **expected case** in the paper (argmax prediction is the expected class).
+ `out.passed.idx`: a list of example indexes that passed the plausibility test.
+ `out.verum`: saliency output for the expected class prediction for **all** test examples.
+ `out.malum`: saliency output for the unexpected class prediction for **all** test examples.

## Evaluate Faithfulness

The faithfulness evaluations will be based on the saliency outputs generated from the step above. To help understand how faithfulness scripts work, we provided example saliency outputs for the `Transformer+SG` configuration on synthetic data in `scripts/faithfulness/example_gender` and `scripts/faithfulness/example_number`, respectively.

Follow the script in `scripts/faithfulness/{input-gender,input-number,model}` to reproduce those numbers.

## Citation

```
@inproceedings{ding-koehn-2021-evaluating,
    title = "Evaluating Saliency Methods for Neural Language Models",
    author = "Ding, Shuoyang  and
      Koehn, Philipp",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.399",
    pages = "5034--5052",
    abstract = "Saliency methods are widely used to interpret neural network predictions, but different variants of saliency methods often disagree even on the interpretations of the same prediction made by the same model. In these cases, how do we identify when are these interpretations trustworthy enough to be used in analyses? To address this question, we conduct a comprehensive and quantitative evaluation of saliency methods on a fundamental category of NLP models: neural language models. We evaluate the quality of prediction interpretations from two perspectives that each represents a desirable property of these interpretations: plausibility and faithfulness. Our evaluation is conducted on four different datasets constructed from the existing human annotation of syntactic and semantic agreements, on both sentence-level and document-level. Through our evaluation, we identified various ways saliency methods could yield interpretations of low quality. We recommend that future work deploying such methods to neural language models should carefully validate their interpretations before drawing insights.",
}
```

## Naming

Tarsius is a genus of tarsiers, small primates native to islands of Southeast Asia. They are known for their huge eyes that take up nearly their entire head, which gather and reflect any speck of light available and **allow them to see in the dark**.
