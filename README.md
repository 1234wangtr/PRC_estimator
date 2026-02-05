## Repository Structure and Usage

### Complexity Estimation (Section 5)

The scripts for estimating the complexities of our attacks in Section 5 are located in:

./calc_complexities

---
### Code to generate PRC Data

The code for generating PRC Data are located at:
LLM: llm_code/llm_encode.py
GIM: gim_code/main.py

### PRC Data

The ZIP files containing JSON data (including keys and codewords of the PRC scheme) are stored in:

./llm_data
./gim_data

---

### Concrete Attacks (Section 6)

The implementations of the concrete attacks described in Section 6 are provided in:

./llm_real_attack
./gim_real_attack

To run the attacks, please follow these steps:

1. Unzip the data files in:
   ./llm_data
   ./gim_data

2. Run one of the following scripts:
   - read_watertext_data.py (for LLM-based watermark attacks)
   - watermark_img_multi_ddim.py (for GIM-based watermark attacks)

These scripts output the detection rates for:
- original codewords,
- inversion codewords, and
- plain (non-watermarked) codewords.
