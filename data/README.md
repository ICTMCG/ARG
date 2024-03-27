# Dataset

## Access

You will be shared the dataset by email after an ["Application to Use the Datasets from ARG for Fake News Detection"](https://forms.office.com/r/eZELqSycgn) has been submitted.

## Description

The dataset contains `train.json`, `val.json`, and `test.json`. In every json file:

- the `content` identifies the content of the post.
- the `label` identifies the veracity of the post, whose value is  `real` or `fake` in  chinese, and `0` or `1` in english.
- the `time` identifies the publised time of the post.
- the `source_id` is the unique id for the post.
- the `td_rationale` is the LLM rationale from perspective specific prompting (textual description).
- the `td_pred` is the prediction of the `td_rationale`.
- the `td_acc` is whether the `td_pred` is correct.
- the `cs_rationale` is the LLM rationale from perspective specific prompting (commonsense).
- the `cs_pred` is the prediction of the `cs_rationale `.
- the `cs_acc` is whether the `cs_pred` is correct.
- the `split` is the split tag of the post.

