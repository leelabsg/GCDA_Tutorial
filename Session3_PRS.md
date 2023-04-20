# Practice Session #3: PRS (April 25, 2023)

In this session, we are going to construct polygenic risk score using PRS-CS. \
References : [PRS-CS github](https://github.com/getian107/PRScs), [PRS-CS paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6467998/). \
The data we are going to use are already preprocessed or downloaded.

### 0. Log in to leelabguest
``` 
ssh leelabguest@147.47.200.192
```

### 1. Connect CPU
``` 
launch-shell 0 720 leelabsg
``` 

### 2. Activate conda environment
``` 
conda activate python3
``` 

### 4. Make directory for practice session in your directory
``` 
mkdir /home/n1/leelabguest/GCDA/usr/YOUR_DIRECTORY/practice_3 
``` 

### 5. Run PRScs 
``` 
python /home/n1/leelabguest/GCDA/3_PRS/PRScs/PRScs.py \
--ref_dir=/home/n1/leelabguest/GCDA/3_PRS/data/reference/ldblk_1kg_eas \
--bim_prefix=/home/n1/leelabguest/GCDA/3_PRS/data/plink/sample \
--sst_file=/home/n1/leelabguest/GCDA/3_PRS/data/summary_stat/sumstats.txt \
--n_gwas=177618 \
--out_dir=/home/n1/leelabguest/GCDA/usr/YOUR_DIRECTORY/practice_3/prscs
``` 

### 6. Merge chr1 - chr22 beta files into one file 
``` 
for i in {1..22}; do cat "/home/n1/leelabguest/GCDA/usr/YOUR_DIRECTORY/practice_3/prscs_pst_eff_a1_b0.5_phiauto_chr$i.txt" >> /home/n1/leelabguest/GCDA/usr/YOUR_DIRECTORY/practice_3/prscs_chr1-22.txt; done
``` 

### 7. Calculate PRS using plink 
``` 
/home/n1/leelabguest/plink \
--bfile /home/n1/leelabguest/GCDA/3_PRS/data/plink/sample \
--score /home/n1/leelabguest/GCDA/usr/YOUR_DIRECTORY/practice_3/prscs_chr1-22.txt 2 4 6 \
--out /home/n1/leelabguest/GCDA/usr/YOUR_DIRECTORY/practice_3/score
``` 
