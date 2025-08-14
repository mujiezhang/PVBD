<div align="center">
  <img src="https://github.com/user-attachments/assets/1e6a744b-1286-49a9-89b0-fe815ea30a35" alt="ProBord" width="450" />
</div>

# ProBord: **Pro**virus **Bord**er Delimiter
ProBord (**Pro**virus **Bord**er Delimiter) is a bioinformatics tool that predicts the precise borders of proviruses by identifying attL/R sites.

# Introduction
## 🧬 Provirus integration process

A provirus usually refers to a virus integrated into a prokaryotic chromosome as a stable genetic element. Before integration, the phage attP site and the host attB site—share core sequence—undergo site-specific recombination catalyzed by integrase (Int), producing attL and attR sites flanking the prophage in the host genome. During excision, attL and attR recombine in reverse, mediated by integrase and excisionase (Xis), restoring attP on the free phage DNA and attB on the host chromosome. Unless otherwise stated, attB, attP, and attL/R refer to their core sequences.

![integration](https://github.com/user-attachments/assets/7795a4b2-fdef-4b7f-8737-99b6bd4be02d)


## 💡 Workflow of ProBord

- Step1: Preprocessing viral region
  - ...
- Step2: Identifing candidate att
  - ...
- Step3: Comparing and scoring
  - ...

![workflow](https://github.com/user-attachments/assets/9cb7005f-0695-4f93-8b55-e4b6428b4d36)

# Instructions

## Dependencies
- ProBord is a Python script that relies on:
  
```
blastn
python3
biopython
checkv=1.0.3
ncbi-genome-download
```

## Installation

- Install miniconda and add channels (If already installed, please skip)
```
wget -c https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda config --add channels bioconda
conda config --add channels conda-forge
```
- Install dependencies
```
conda create -n probord checkv=1.0.3 ncbi-genome-download
git clone https://github.com/mujiezhang/ProBord.git
conda activate probord
```

## Database preparation
- Prepare the CheckV database (if needed; otherwise skip):  `checkv download_database ./ `
- Prepare blastn database for attB detection (**required**):
  - If your provirus originates from a specific bacterial/archaeal genus, you only need to download bacterial/archaeal genomes and create a blastn database for that genus using the script `prepare_blastn_db.sh`. For example, for the genus "Mannheimia": `bash prepare_blast_db.sh Mannheimia bacteria`.
  - However, downloading large genera like `Escherichia` can be extremely time-consuming since they contain tens of thousands of genomes. Therefore, we have prebuilt databases for NCBI RefSeq (release 228) genera with over 5,000 genomes, including ``;``;``;``. You can download these prebuilt databases using the script `prepare_blastn_db.sh` by specifying the genus and the "prebuild" parameter. Note that downloading these prebuilt databases still requires significant time due to the large genome counts.
  - If you have numerous proviruses from diverse genera, or if you don't know your provirus host classification, you can download the NCBI nt database or all bacterial/archaeal genomes from RefSeq, create a blastn database, and then run probord. (This approach consumes substantial storage space and memory, and will significantly increase probord's runtime.)

mkdir Mannheimia_db; cd Mannheimia_db
ncbi-genome-download -F fasta --genera Mannheimia -o Mannheimia --flat-output  -P   -p 4 bacteria
gunzip Mannheimia/*.gz
cat Mannheimia/*.fna > Mannheimia.fna
makeblastdb -in Mannheimia.fna -dbtype nucl -out Mannheimia
rm -rf Mannheimia;rm Mannheimia.fna
```
## Input files

## How to run

## Output files

# useful emoji 💡🧬🔆✅🎉🚀🚩⌛⚙️🖋️

# Citation
......

# 📬 Contact
```
# Mujie Zhang
# School of Life Sciences & Biotechnology, Shanghai Jiao Tong University
# Email: zhangmujie@sjtu.edu.cn
```
