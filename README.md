## Repository Structure and Usage

### Complexity Estimation (Section 5)

The scripts for estimating the complexities of our attacks in Section 5 are located in `calc_complexities` directory.

The estimation can be reproduced by running the following commands:
```
cd calc_complexities
python llm_watermark_secrity_estim.py # Figure 4(a), Table 6
python gim_watermark_secrity_estim.py # Figure 4(b), Table 7
```
This will reproduce the following results:
- `calc_complexities\llm_watermark_secrity_estim.pdf` (Figure 4(a))
- `calc_complexities\gim_watermark_secrity_estim.pdf` (Figure 4(b))
- `calc_complexities\llm_watermark_secrity_estim.csv` (Table 6)
- `calc_complexities\gim_watermark_secrity_estim.csv` (Table 7)

### Code to generate PRC Data

The code for generating PRC Data are located at:
LLM: llm_code/llm_encode.py
GIM: gim_code/main.py


### Watermark Scheme Implementation and PRC Data Generation (Section 6)

Our watermark schemes for LLMs and GIMs are implemented in the following directories:

```
./llm_code
./gim_code
```

To generate the JSON data used in our attacks (including watermark-related PRC keys and codewords), you can run the following scripts:

- **LLM-based watermark generation:**
  ```
  ./llm_code/llm_encode.py
  ```

- **GIM-based watermark generation:**
  ```
  ./gim_code/main.py
  ```

Running the watermark generation scripts may require access to the corresponding model files, such as **DeepSeek** for LLMs or **Stable Diffusion** for GIMs.

---

### LLM Quality (Section 6.2.1)
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
