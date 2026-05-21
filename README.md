# Cryptanalysis of Pseudorandom Error-Correcting Codes

This is the official artifacts repository for the paper "Cryptanalysis of Pseudorandom Error-Correcting Codes" (accepted by USENIX Security 2026 at Cycle 2 with Paper ID #1773).

In this paper, we analyze the security of pseudorandom error-correcting codes (PRCs), and propose three attacks against their undetectability and robustness properties.
Since PRCs are used for watermarking generative content, our attacks can be used to detect and remove those watermarks, revealing the concrete security limits of PRCs in real-world watermarking applications.

This artifact contains the code for both thoretical analysis of our attacks, along with the attack against real-world PRC watermark schemes for Large Language Models (LLMs) and Generative Image Models (GIMs). We also provide the example data for reproducing the results in our paper.

## System Requirements
- Software: an Linux-based environment with `conda`, `python3`, `pip`, and `git`, `wget`, and `unzip` installed.
- Hardware: Generating PRC watermarked content requires a GPU with at least 80GB of VRAM. However, the attacks and result analysis can be reproduced on a CPU-only machine with the provided data artifacts.


## Setup Instructions
- Download the artifacts from [Zenodo](https://zenodo.org/records/20321793) or directly clone the repository from [GitHub](https://github.com/1234wangtr/PRC_estimator).
- Unzip the example data artifacts
   ```bash
   cd prc-estimator
   unzip gim/data/gen_data_SD21_t3.zip -d gim/data/gen_data_SD21_t3
   unzip llm/data/gen_result_Deepseek_t_3_temp_1.8.zip -d llm/data/gen_result_Deepseek_t_3_temp_1.8
   unzip llm/data/gen_result_Deepseek_t_3_temp_all.zip -d llm/data
   ```
- Create the conda environment for llm and gim.
   ```bash
   conda create -n prc-estimator-llm python=3.11 -y
   conda activate prc-estimator-llm
   pip install -r llm/requirements.txt

   conda create -n prc-estimator-gim python=3.11 -y
   conda activate prc-estimator-gim
   pip install -r gim/requirements.txt
   ```
- Download the publicly available models and datasets.
  - [Stable Diffusion 2.1 Base](https://huggingface.co/Manojb/stable-diffusion-2-1-base) (For GIM main experiments)
  - [DeepSeek Qwen2.5 8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) (For LLM main experiments)
  - [Stable Diffusion 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) (For GIM ablation experiments)
  - [Stable Diffusion 2 Base](https://huggingface.co/Manojb/stable-diffusion-2-base) (For GIM ablation experiments)
  - [Qwen3 8B](https://huggingface.co/Qwen/Qwen3-8B) (For LLM ablation experiments)
  - [Stable Diffusion Prompts](https://huggingface.co/Gustavosta/Stable-Diffusion-Prompts)

## Project Structure
The artifacts are seperated into two main folders: `llm` for the LLM-based experiments and `gim` for the GIM-based experiments. Each folder is organized as follows:
```
llm / gim
├── requirements.txt # Python dependencies for the experiments 
├── security_estim # Code for estimating the time complexities of our attacks
├── generation # Code for generating PRC watermarked content
├── data # Example watermarked content and necessary data for reproducing the attack results
└── attack # Code for the concrete attacks
```

## Concrete Time Complexity Analysis (Section 5, Figure 4, Table 6 and Table 7)

The scripts for estimating the complexities of our attacks in Section 5 are located in `<gim or llm>/security_estim/main.py` directory.

The estimation can be reproduced by running the following commands:
```bash
conda activate prc-estimator-llm && python llm/security_estim/main.py # Figure 4(a), Table 6
conda activate prc-estimator-gim && python gim/security_estim/main.py # Figure 4(b), Table 7
```

This will reproduce the following results:
- `llm/data/llm_watermark_secrity_estim.pdf` (Figure 4(a))
- `gim/data/gim_watermark_secrity_estim.pdf` (Figure 4(b))
- `llm/data/llm_watermark_secrity_estim.csv` (Table 6)
- `gim/data/gim_watermark_secrity_estim.csv` (Table 7)

## Attack Real-World PRC Watermark Schemes (Section 6)

### PRC Watermark for LLMs

#### Watermarked Content Generation (Section 6.1)

To generate the watermarked texts, you can run the following scripts:
```bash
conda activate prc-estimator-llm
python llm/generation/main.py --prc_t 3 --model_name Deepseek --temperature 1.8 --gen_start 0 --gen_end 128
python llm/generation/main.py --prc_t 4 --model_name Deepseek --temperature 1.8 --gen_start 0 --gen_end 128
```

This will generate watermarked texts and the associated PRC data in the `data/llm/gen_result_*` directory.

#### Concrete Attacks (Section 6.2, Table 3)
The implementations of the Attack I and Attack II are provided in `llm/attack`.

To run the attacks and reproduce Table 3, you can run the following scripts:
```bash
conda activate prc-estimator-llm
python llm/attack/attack1_2_main.py
```

This will output the corresponding data for Table 3 in the terminal.

#### Watermarked Content Quality Analysis (Section 6.2.1, Figure 5, and Table 9)
The LLM-generated watermarked and non-watermarked content under different temperatures are unzipped to `llm/data/gen_result`.

To reproduce Figure 5, you can run the following command:
```bash
conda activate prc-estimator-llm
python llm/generation/plot_entropy.py
```
This will generate the following figure:
- `llm/data/avg_entropy_vs_correct_rate.pdf` (Figure 5)

To reproduce Table 9, you can manually check the following JSON files:
- `llm/data/gen_result/temperature_1.0/1748357173909375233.json`
- `llm/data/gen_result/temperature_1.2/1748402150078932357.json`
- `llm/data/gen_result/temperature_1.4/1748402173162096394.json`
- `llm/data/gen_result/temperature_1.6/1748402254743413050.json`
- `llm/data/gen_result/temperature_1.8/1748402215940443270.json`


### PRC Watermark for GIMs

#### Watermarked Content Generation (Section 6.1)
To generate the watermarked images, you can run the following scripts:
```bash
conda activate prc-estimator-gim
python gim/generation/main.py --gen_model_id SD21 --inv_model_ids SD15,SD2,SD21 --prc_t 3 --start 0 --end 128
python gim/generation/main.py --gen_model_id SD21 --inv_model_ids SD15,SD2,SD21 --prc_t 4 --start 0 --end 128
```

This will generate watermarked images and the associated PRC data in the `gim/data/gen_data*` directory.

#### Concrete Attacks (Section 6.3, Table 4, Table 8)
The implementations of the Attack I and Attack II are provided in `gim/attack`.
To run the attacks and reproduce Table 4, you can run the following scripts:
```bash
conda activate prc-estimator-gim
python gim/attack/attack1_2_main.py
```
This will output the corresponding data for Table 4 in the terminal.

To reproduce Table 8, you can run the following scripts:
```bash
conda activate prc-estimator-gim
python gim/attack/attack1_2_diff_inv_main.py
```
This will output the corresponding data for Table 8 in the terminal.

### Watermark Removal Content Quality Analysis (Added in Rebuttal)
Since Attack III requires significantly more computational resources, we directly extract imtermedia results of PRC and only test whether Attack III will degrade the image quality. The data extraction and watermark removal can be reproduced by running the following scripts:

```bash
conda activate prc-estimator-gim
python gim/generation/main-for-attack3.py  # Extract the intermediate data for Attack III
python gim/attack/attack3_main.py  # Remove the watermark using results from Attack III
python gim/attack/attack3_stati.py  # Calculate the attack statistics
```
This will output the success rate of watermark removal under different image quality budgets.


## Acknowledgements
The code for generating PRC watermarked images is adapted from https://github.com/XuandongZhao/PRC-Watermark.