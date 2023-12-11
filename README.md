# Google's TRC Program: A Short Guide


Google Research TPU Research Cloud is a cluster of 1,000 Cloud TPUs made available free of charge to researchers from around the world.
[Applications](https://sites.research.google/trc/about/) are open year round and approved on a rolling basis.

TRC currently offers researchers access to the following resources:

    100 preemptible Cloud TPU v2-8 device(s) in zone us-central1-f
    5 on-demand Cloud TPU v3-8 device(s) in zone europe-west4-a
    5 on-demand Cloud TPU v2-8 device(s) in zone us-central1-f

**Expectations**

Participants are expected to "share their TRC-supported research with the world through peer-reviewed publications, open source code, blog posts, or other means."

**How to Cite TRC**

Use "Research supported with Cloud TPUs from Google's TPU Research Cloud (TRC)" or similar.

**Compute Intensive Projects**

Larger projects can submit a research proposal requesting more compute, as mentioned in the FAQ:

    Our goal is to accelerate open machine learning research.
    If you have a proposal for using large amounts and/or specific generations of Cloud TPU devices please contact us with additional information about your project's goals, needs, and timeline.
    For example, the open source MiniGo project successfully used 640 Cloud TPUs simultaneously via GKE.


## Training Large Models on Cloud TPU

Setting up a TPU VM is fairly straightforward and is well covered in Google's documentation (see https://cloud.google.com/tpu/docs/run-calculation-jax, for an introduction).

A training workflow typically goes something like this:
1. spin up a TPU VM in a given zone
2. ssh into all workers and install software
3. download datasets and configuration files or (more likely) upload both to Google Cloud Storage
4. train the model
5. wait 
6. save model weights and other artifacts
7. delete your Cloud TPU

Much of this is boilerplate, so opting to use a framework is probably a good idea.
Here are a few:
[Levanter](https://github.com/stanford-crfm/levanter/tree/main),
[EasyLM]( https://github.com/young-geng/EasyLM/tree/main), 
[JAXSeq]( https://github.com/Sea-Snell/JAXSeq), 
[Mesh Transformer Jax]( https://github.com/kingoflolz/mesh-transformer-jax),
[Paxml (aka Pax)](https://github.com/google/paxml/tree/main),
[FLAX](https://github.com/google/flax/tree/main/examples/lm1b),and 
[Hugging Face's Transformers](https://github.com/huggingface/transformers/tree/main/examples/flax)

Levanter is a newly developed JAX based library for training foundation models.

It is constructed on top of a named tensor library, Haliax, and supports most of what we'll need when training large models: data preprocessing, sharded data loading, WandB integration, distributed checkpointing, Hugging Face Hub integration and so on.

It also happens to be exceptionally well documented, making it easy to adapt to particular use cases.


## Using Stanford's Levanter Framework
Use the cli to log in (see Gcloud CLI section below for a list of commands) then export the following variables (wandb and hugging face are optional):

```
export VM_NAME=<your_vm_name>
export ZONE=<your_zone>          
export TPU_TYPE=<your_tpu_type>  
export BUCKET_NAME=<gs://your_gs_bucket_path>
export WANDB_API_KEY=<your_wandb_api_key>
export HUGGING_FACE_HUB_TOKEN=<your_hf_token>
```

Next, spin up a VM:
```
bash infra/spin-up-vm.sh $VM_NAME -z $ZONE -t $TPU_TYPE
```

If you are using your own config, upload it to GCS:
```bash
gsutil cp my_config.yaml gs://my_bucket//my_config.yaml
```
To restart a training run in the event of a crash or if a preemptible VM is preempted, it's advisable to use the  babysitting script:
```
infra/babysit-tpu-vm <name> -z <zone> -t <type> [--preemptible] -- \
    WANDB_API_KEY=... levanter/infra/run.sh python levanter/src/levanter/main/train_lm.py --config_path gs://my_bucket/my_config.yaml \
    --trainer.checkpointer.base_path gs://path/to/checkpoints/

```

Datasets  can either be pulled directly from Hugging Face Hub or from GCS: 
```yaml
# levanter/config/llama2_7b.yaml
data:
  train_urls:
    - "gs://pubmed-mosaic/openwebtext-sharded/openwebtext_train.{1..128}-of-128.jsonl.gz"
  validation_urls:
    - "gs://pubmed-mosaic/openwebtext-sharded/openwebtext_val.{1..8}-of-8.jsonl.gz"
  cache_dir: "gs://levanter-data/tokenized/openwebtext_llama/"
  tokenizer: "meta-llama/Llama-2-70b-hf"
model:
  type: llama
trainer:
....
#
```
```yaml
# levanter/examples/gsm8k-lora/gsm8k-llama2.yaml
model_name_or_path: "meta-llama/Llama-2-7b-hf"
data: gsm8k
data_cache_dir: gsm8k_cache
trainer:
....
# 
```

For extra logging or to connect to tensorboard, we can use port forwarding,
```bash
gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE -- \
    -L 6006:localhost:6006

```
then from a separate terminal window or tmux pane (i.e. while still logged in), connect to tensorboard:

```
tensorboard --logdir=$HOME/logs
```



## Sans Framework
Without a framework, we need a few additional steps.

See https://cloud.google.com/tpu/docs/v5e-training#train-gpt2-on-the-oscar-dataset for more information (code below adapted from same)

First, set the version variable (called VM_IMAGE here) to 'tpu-vm-base' for v2 and v3 TPU versions or 'tpu-vm-v4-base' for v4 TPUs etc.

```
#  include the os image version and project id:
export VM_IMAGE=tpu-vm-base
export PROJECT_ID=<your_project_name> 
```
Then spin up a machine:
```
gcloud compute tpus tpu-vm create $VM_NAME \
  --zone=$ZONE \
  --accelerator-type=$TPU_TYPE \
  --version=$VM_IMAGE" \
  --preemptible 
```
Here we add the current vm's ssh keys:
```
ssh-add ~/.ssh/google_compute_engine
```

We can now install software and/ or download configs:
```
gcloud compute tpus tpu-vm ssh ${VM_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command='git clone https://github.com/huggingface/transformers.git && cd transformers && pip install . && pip install -r examples/flax/_tests_requirements.txt && pip install --upgrade huggingface-hub urllib3 zipp && pip install tensorflow && pip install -r examples/flax/language-modeling/requirements.txt'

gcloud compute tpus tpu-vm ssh ${VM_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command='cd transformers/examples/flax/language-modeling && gsutil cp -r gs://cloud-tpu-tpuvm-artifacts/v5litepod-preview/jax/gpt .'

```
Once completed, we can finally run the training script:
```
# Train the model with a pre-mapped buffer at 4GB.
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} 
  --project=${PROJECT_ID} 
  --zone=${ZONE} 
  --worker=all 
  --command='cd transformers/examples/flax/language-modeling && TPU_PREMAPPED_BUFFER_SIZE=4294967296 JAX_PLATFORMS=tpu python3 run_clm_flax.py --output_dir=./gpt --model_type=gpt2 --config_name=./gpt --tokenizer_name=./gpt --dataset_name=oscar --dataset_config_name=unshuffled_deduplicated_no --do_train --do_eval --block_size=512 --per_device_train_batch_size=4 --per_device_eval_batch_size=4 --learning_rate=5e-3 --warmup_steps=1000 --adam_beta1=0.9 --adam_beta2=0.98 --weight_decay=0.01 --overwrite_output_dir --num_train_epochs=3 --logging_steps=500 --eval_steps=2500'
```

## Helpful Linux & Google Utilities 

### gcloud 
```
## check logged in status and correct project
$ gcloud config list
> [core]
> account = your@gmail.com
> disable_usage_reporting = False
> project = llama-2-gsm8k

## list all active vms in a given zone:
$ gcloud compute instances list  --zone=$ZONE
> Listed 0 items.

## SSH into TPU VM:
$ gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE 

## Upload config file to all tpu worker machines via scp
# scp: FROM(relative path) TO (tpu name/absolute path)
gcloud compute tpus tpu-vm scp --recurse config/data/openwebtext_source.yaml $VM_NAME:/home/x/levanter/config/data/ --zone $ZONE --worker=all



```


### tmux
    "Ctrl+B ?"  View ALL keybindings. Press Q to exit.

    "Ctrl+B D"  Detach from the current session.
    
    # To scroll using mouse
    "Ctrl+B :"  then type in "set -g mouse"

### ssh-add (local)
ssh-add /Users/v/.ssh/google_compute_engine

### Parallel Shells

**PDSH**

To get pdsh to work with MacOS, we first have to export an argument to skip strict host file checking. Once that's done, we create a 'hosts' text file from which pdsh can read hosts ips and names.
```
# see https://gist.github.com/pm-hwks/17bbea3956a2e4efd135b396c01e9da8
# Skips strict host file checking which does not work on macos out-of-the-box (can be added to .bashrc or .zshrc)

export PDSH_SSH_ARGS_APPEND="-q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PreferredAuthentications=publickey"


# see: levanter_workspace/levanter/docs/Getting-Started-TPU-VM.md
# save all the worker ips to a "my-hosts" file
gcloud compute tpus tpu-vm describe --zone us-east1-d $name | awk '/externalIp: (.*)/ {print $2}'  > my-hosts

# now we can run any command line arguments across all machines in parallel
pdsh -R ssh -w ^my-hosts  "mkdir -p ~/alpaca_dataset & mkdir -p ~/alpaca_cache"


```

There are other utilities with similar functionality. _mosh_, _ppsh_, etc. 
For example, here is how **PPSH** does rsync and pkill across all workers:
```bash
# prsync copying of files:
prsync -h ~/.pssh_hosts_files /etc/passwd /tmp/
prsync -h ~/.pssh_hosts_files *.html /var/www/html/

# killing processes in parallel on N hosts
pnuke -h ~/.pssh_hosts_files firefox
pnuke -h ~/.pssh_hosts_files nginx
```

# Training Examples
This section shows two examples of end-to-end training using Levanter: pre-training a  **GPT-2** model from scratch and fine-tuning a **Llama-2** model using a custom  dataset.

Also included are benchmarks for the alpaca model. Benchmarking was done using the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) protocol. Open LLM leaderboard's backend runs the Eleuther AI Language Model Evaluation Harness. Specifically, they use the following 7 benchmarks and num_fewshot:

    AI2 Reasoning Challenge (25-shot) - a set of grade-school science questions.
    HellaSwag (10-shot) - a test of commonsense inference, which is easy for humans (~95%) but challenging for SOTA models.
    MMLU (5-shot) - a test to measure a text model's multitask accuracy. The test covers 57 tasks including elementary mathematics, US history, computer science, law, and more.
    TruthfulQA (0-shot) - a test to measure a model's propensity to reproduce falsehoods commonly found online. Note: TruthfulQA in the Harness is actually a minima a 6-shots task, as it is prepended by 6 examples systematically, even when launched using 0 for the number of few-shot examples.
    Winogrande (5-shot) - an adversarial and difficult Winograd benchmark at scale, for commonsense reasoning.
    GSM8k (5-shot) - diverse grade school math word problems to measure a model's ability to solve multi-step mathematical reasoning problems.

**Note: Open LLM Leaderboard uses the "b281b0921b" branch of EleutherAI/lm-evaluation-harness to get their results (see 'About' section). The same branch was used for alpca evaluations below.**
## Pre-Training GPT-2





Notes:
Karpathy reproduced GPT-2 (124M) on OpenWebText, on a single 8XA100 40GB node in about 4 days of training. Using flash attention (the first iteration of naonGPT came out before flash attn.) should considerably speed this up, however.

Levanter reaches a validation loss of ~2.88 in about 30 hour (gpt2_small.yaml).
Based on a few short test runs with gpt2_small_fast.yaml, we can probably get that down to ~19 hours or so.


| Repository | Time | Steps | Val. Loss |                                 |
|------------|------|-------|-----------|---------------------------------|
|naoGPT      |~4days| 399k  |   2.95    | Torch DDP, no flash attention
|levanter    |~30hrs| 300k  |   2.88    | gpt2_small.yaml                 |
|levanter    |      |       |           | gpt2_small_fast.yaml            |


![image](https://github.com/baricev/google-trc-first-steps/assets/153478676/30840193-dfcd-47c6-8940-ed85beabdd3f)

## Fine-Tuning Alpaca Llama-2-7B

Notes:

Dataset used:

levanter-specific, built with data from the [Instruction Tuning with GPT-4](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/) repository. 

[alpaca_gpt4_data.json](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data.json) contains 52K instruction-following data generated by GPT-4 with prompts in Alpaca. 


Converted to jsonl and uploaded as a Hugging Face dataset to HF hub ( 50% reduction in dataset size).



Tested the eval harness with Llama-2-7b as a sanity-check (numbers match those published by Open LLM).

Results for arc_challenge(25 shot) and hellaswag (10 shot) show below (gpu only!)

![image](https://github.com/baricev/google-trc-first-steps/assets/153478676/4f115fbb-1504-41c0-95b4-96cacc8a0773)


## Llama-2-7b Benchmarks
#### Published Open LLM Leaderboard Results
| Model    | Task          | Version | Metric   | Value  | ± | Stderr |
|----------|---------------|---------|----------|--------|---|--------|
| Llama2-7 | arc_challenge | 25      | acc      | 0.4923 | ± | 0.01   |
| (meta)   |               |         | acc_norm | 0.5307 | ± | 0.0146 |
|          | hellaswag     | 10      | acc      | 0.5882 | ± | 0.0049 |
|          |               |         | acc_norm | 0.7855 | ± | 0.0041 |



#### Test Results
| Model    | Task          | Version | Metric   | Value  | ± | Stderr |
|----------|---------------|---------|----------|--------|---|--------|
| Llama2-7 | arc_challenge | 25      | acc      | 0.4915 | ± | 0.0146 |
| (meta)   |               |         | acc_norm | 0.5299 | ± | 0.0146 |
|          | hellaswag     | 10      | acc      | 0.5878 | ± | 0.0049 |
|          |               |         | acc_norm | 0.7859 | ± | 0.0041 |


## Alpaca Benchmarks
#### Open LLM Leaderboard 
No published results found for Llama-2-7B Alpaca models.

#### Test Results
| Model    | Task          | Version | Metric   | Value  |   | Stderr |
|----------|---------------|---------|----------|--------|---|--------|
|Llama2-7  | arc_challenge | 25      | acc      | 0.5179 | ± | 0.0146 |
|(alpaca)  |               |         | acc_norm | 0.5478 | ± | 0.0145 |
|          | hellaswag     | 10      | acc      | 0.6030 | ± | 0.0049 |
|          |               |         | acc_norm | 0.7907 | ± | 0.0041 |


---
For some context :)
![image](https://github.com/baricev/google-trc-first-steps/assets/153478676/44701def-e2f3-464b-80cc-0e4befb2c6b3)
