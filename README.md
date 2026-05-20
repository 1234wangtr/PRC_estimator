# Cryptanalysis of Pseudorandom Error-Correcting Codes

This is the official artifacts repository for the paper "Cryptanalysis of Pseudorandom Error-Correcting Codes" (accepted by USENIX Security 2026 at Cycle 2 with Paper ID #1773).

In this paper, we analyze the security of pseudorandom error-correcting codes (PRCs), and propose three attacks against their undetectability and robustness properties.
Since PRCs are used for watermarking generative content, our attacks can be used to detect and remove those watermarks, revealing the concrete security limits of PRCs in real-world watermarking applications.

This artifact contains the code for both thoretical analysis of our attacks, along with the attack against real-world PRC watermark schemes for Large Language Models (LLMs) and Generative Image Models (GIMs). We also provide the example data for reproducing the results in our paper.

## Prerequisites
- Software: an Linux-based environment with `conda`, `python3`, `pip`, and `git`, `wget`, and `unzip` installed.
- Hardware: Generating PRC watermarked content requires a GPU with at least 80GB of VRAM. However, the attacks and result analysis can be reproduced on a CPU-only machine with the provided data artifacts.


## Setup Instructions
- Download the code artifacts from GitHub or Zenodo.
```
git clone https://github.com/yourusername/prc-estimator && cd prc-estimator # from GitHub
wget https://zenodo.org/record/1234567/files/prc-estimator.zip && unzip prc-estimator.zip && cd prc-estimator # from Zenodo
```
- Unzip the data artifacts from GitHub or Zenodo.
```
unzip llm_data.zip -d llm_data
unzip gim_data.zip -d gim_data
```
- Create a conda environment and install the required dependencies.
```
conda create -n prc-estimator python=3.12 -y
conda activate prc-estimator
pip install -r requirements.txt
```
- Download public GIM and LLM models and prompt datasets.
  - Stable Diffusion 2.1 Base:
  - Stable Diffusion 2.1 Base Inversion:
  - DeepSeek Qwen2.5 8B:
  - Qwen3 8B: 

## Concrete Time Complexity Analysis (Section 5)

The scripts for estimating the complexities of our attacks in Section 5 are located in `calc_complexities` directory.

The estimation can be reproduced by running the following commands:
```
cd calc_complexities
conda activate prc-estimator
python llm_watermark_secrity_estim.py # Figure 4(a), Table 6
python gim_watermark_secrity_estim.py # Figure 4(b), Table 7
```

This will reproduce the following results:
- `calc_complexities\llm_watermark_secrity_estim.pdf` (Figure 4(a))
- `calc_complexities\gim_watermark_secrity_estim.pdf` (Figure 4(b))
- `calc_complexities\llm_watermark_secrity_estim.csv` (Table 6)
- `calc_complexities\gim_watermark_secrity_estim.csv` (Table 7)

## Attack Real-World PRC Watermark Schemes (Section 6)

### PRC-watermarked Content Generation (Section 6.1)

The code for generating PRC watermarked texts (LLMs) and images (GIMs) is provided in `./llm_code` and `./gim_code` directories, respectively.

To generate the watermarked texts, you can run the following scripts:
```
cd llm_code
conda activate prc-estimator
python llm_encode.py
```

The generated watermarked texts are stored in `./llm_data/gen_result` directory.
We also provide the generated watermarked texts in `./llm_data/gen_result.zip` for reproducing the results without running the generation scripts.


To generate the watermarked images, you can run the following scripts:
```
cd gim_code
conda activate prc-estimator
python main.py
```

This script will only store the necessary watermark data in `./gim_data/watermark_data` without storing the generated watermarked images.
We also provide the generated watermark data in `./gim_data/watermark_data.zip` for reproducing the results without running the generation scripts.

### LLM Quality Analysis (Section 6.2.1)
The LLM-generated watermarked and non-watermarked content are zipped in `./llm_data/gen_result`.

To reproduce Figure 5, you can run the following command:
```
python llm_code/plot_entropy.py
```

To check Table 9, you can manually check the following JSON files:
- `llm_data\gen_result\temperature_1.0\1748357173909375233.json`
- `llm_data\gen_result\temperature_1.2\1748402150078932357.json`
- `llm_data\gen_result\temperature_1.4\1748402173162096394.json`
- `llm_data\gen_result\temperature_1.6\1748402254743413050.json`
- `llm_data\gen_result\temperature_1.8\1748402215940443270.json`

### Concrete Attacks (Section 6.2)

The implementations of the concrete attacks described in Section 6 are provided in:

./llm_real_attack
./gim_real_attack_normal
./gim_real_attack_diff_inv

To run the attacks, please follow these steps:

1. Unzip the data files in:
   ./llm_data
   ./gim_data

2. Run one of the following scripts:
   - read_watertext_data.py (for LLM-based watermark attacks I & II)
   - read_waterimg_data.py (for GIM-based watermark attacks I & II)
   - watermark_img_multi_ddim.py (for GIM-based watermark attacks I & II)
   - gim_real_attack\attack3-adv.py (for GIM-based watermark attack III)

These scripts output the detection rates for:
- original codewords,
- inversion codewords, and
- plain (non-watermarked) codewords.


## Acknowledgements
The code for generating PRC watermarked images is adapted from https://github.com/XuandongZhao/PRC-Watermark.