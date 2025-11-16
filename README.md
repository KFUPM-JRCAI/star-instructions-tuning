# instructions-tuning

This project aims to tune LLMs on Arabic datasets.

# Project Strucutre

- The folder `jrcai_corekit` is a set of python utiliites developed internally in the center (by Eng. Raed Mughaus, repo: https://github.com/MagedSaeed/llms-corekit) to make interactions with LLMs either on inference or tuning much easier.
- Folders `eval_harness_extra_tasks` and `experimental_hf_datasets` are axulary directories used during evaluation. `eval_harness_extra_tasks` contains tasks descriptions written in yaml so that eval-harness understands how to perform evaluation. `experimental_hf_datasets` is a folder containing custom constructed huggingface datasets saved in parquet format. Useful for LLMs evaluation as well.

- The `Notebooks` folder contains the experiments and the fine-tuned models codes and files. The folder `cross_tasks_tuning` is an experimental folder and can be ignored. Similarly, `playground_experimentation` contains some experimentation code that can be ignored. Other folders [`dialect_identification`,`machine_translation`,`NLI`,`NLU`,`sarcasm_detection`,`summarization`] contains the experiments performed for different NLP tasks and mainly reported in chapter 5 in the usecase report. Each of these folder has mainly the following structure: 
```
{task}/
├── {dataset_name}/
│   └── {model_name}/
│       ├── evaluate/
│       │   └── {model_version}.ipynb
|       |   ...
│       ├── tune.ipynb
│       ├── tuned_models/
│       |   └── {model_name_version}
|       |  ...
```
Dots, in this tree strcuture, means that other folders can be added on top of this strcture mainly for experimentation. For instance, some tasks contains `Notebooks` folder for further experimentation.

- The strucutre of each evaluation notebook is that we first get the prompts related to the dataset from Tajeeh. After that, we merge the prompts with the dataset instances. Then, we create a yaml task description suitable to be read by eval-harness along with a huggingface dataset for each prompt. Finally, we call eval-harness to perform the evaluation and report the results. Results and evaluation logs are reported and saved in the folder `evaluation_results`.